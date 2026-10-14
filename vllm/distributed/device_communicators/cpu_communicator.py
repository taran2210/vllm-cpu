# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import os
import socket
from typing import Any

import torch
from torch.distributed import ProcessGroup

from vllm.distributed.utils import pickle
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.platforms.interface import CpuArchEnum

from .base_device_communicator import DeviceCommunicatorBase

logger = init_logger(__name__)


class CpuCommunicator(DeviceCommunicatorBase):
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
        use_all2all: bool = False,
    ):
        super().__init__(
            cpu_group, device, device_group, unique_name, use_all2all=use_all2all
        )
        self.dist_module = torch.distributed

        # dp/ep carry the MoE dispatch/combine traffic under data parallel
        use_shm_group = unique_name.startswith(("tp", "pp", "dp", "ep"))
        shm_group_name = None
        if (
            (
                current_platform.get_cpu_architecture() == CpuArchEnum.X86
                or current_platform.get_cpu_architecture() == CpuArchEnum.ARM
                or current_platform.get_cpu_architecture() == CpuArchEnum.POWERPC
                or current_platform.get_cpu_architecture() == CpuArchEnum.S390X
            )
            and hasattr(torch.ops._C, "init_shm_manager")
            and use_shm_group
        ):
            shm_group_name = self._agree_on_shm_group_name()
        if shm_group_name is not None:
            self.dist_module = _CPUSHMDistributed(self, shm_group_name)
        elif use_shm_group:
            logger.info(
                "CPU SHM communicator disabled for group %s: ranks are not on "
                "the same host, falling back to torch.distributed.",
                unique_name,
            )

        # send/recv tensor_dict is only supported through the SHM communicator backend
        self.supports_tensor_dict = isinstance(self.dist_module, _CPUSHMDistributed)

        if self.use_all2all:
            if self.all2all_backend not in (
                "naive",
                "allgather_reducescatter",
            ):  # type: ignore[has-type]
                logger.warning(
                    "`%s` all2all manager is not supported on CPU. "
                    "Falling back to `allgather_reducescatter` manager.",
                    self.all2all_backend,  # type: ignore[has-type]
                )
            from .all2all import AgRsAll2AllManager

            self.all2all_manager = AgRsAll2AllManager(self.cpu_group)
            logger.info("Using allgather_reducescatter all2all manager.")

    def _agree_on_shm_group_name(self) -> str | None:
        """CPUSHM requires all ranks in this group to be on one host and to agree
        on one SHM group name. Groups spanning DP engines (e.g. dp/ep) derive
        different names from their engine-local VLLM_DIST_IDENT, so every rank
        adopts the group leader's name. Returns None if ranks span hosts.
        """
        local = (_CPUSHMDistributed.make_group_name(self), socket.gethostname())
        gathered: list[tuple[str, str]] = [("", "")] * self.world_size
        torch.distributed.all_gather_object(
            gathered,
            local,
            group=self.device_group,
        )
        if len({host for _, host in gathered}) != 1:
            return None
        return gathered[0][0]

    def _dist_module_for(self, input_: torch.Tensor) -> Any:
        # SHM collectives only dispatch floating types.
        if input_.is_floating_point():
            return self.dist_module
        return torch.distributed

    def all_reduce(self, input_):
        self._dist_module_for(input_).all_reduce(input_, group=self.device_group)
        return input_

    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None:
        """NOTE: We assume that the input tensor is on the same device across
        all the ranks.
        NOTE: `dst` is the local rank of the destination rank.
        """
        world_size = self.world_size
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Allocate output tensor.
        if self.rank_in_group == dst:
            gather_list = [torch.empty_like(input_) for _ in range(world_size)]
        else:
            gather_list = None

        # Gather.
        self.dist_module.gather(
            input_, gather_list, dst=self.ranks[dst], group=self.device_group
        )

        if self.rank_in_group == dst:
            output_tensor = torch.cat(gather_list, dim=dim)
        else:
            output_tensor = None
        return output_tensor

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        input_size = input_.size()
        # NOTE: we have to use concat-style all-gather here,
        # stack-style all-gather has compatibility issues with
        # torch.compile . see https://github.com/pytorch/pytorch/issues/138795
        output_size = (input_size[0] * self.world_size,) + input_size[1:]
        # Allocate output tensor.
        output_tensor = torch.empty(
            output_size, dtype=input_.dtype, device=input_.device
        )
        # All-gather.
        self._dist_module_for(input_).all_gather_into_tensor(
            output_tensor, input_, group=self.device_group
        )

        # Reshape
        output_tensor = output_tensor.reshape((self.world_size,) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(
            input_size[:dim]
            + (self.world_size * input_size[dim],)
            + input_size[dim + 1 :]
        )
        return output_tensor

    def _all_gather_dim0(self, input_: torch.Tensor) -> torch.Tensor:
        input_ = input_.contiguous()
        if input_.numel() == 0:
            return input_.new_empty((0,) + input_.shape[1:])
        return self.all_gather(input_, dim=0)

    def _all_gatherv_padded(
        self, input_: torch.Tensor, sizes: list[int] | None
    ) -> torch.Tensor:
        if sizes is None or all(s == sizes[0] for s in sizes):
            return self._all_gather_dim0(input_)
        # Pad every rank to the largest size, gather, then drop padding.
        assert input_.shape[0] == sizes[self.rank_in_group]
        max_size = max(sizes)
        padded = input_.new_empty((max_size,) + input_.shape[1:])
        padded[: input_.shape[0]].copy_(input_)
        gathered = self._all_gather_dim0(padded)
        return torch.cat(
            [gathered[r * max_size : r * max_size + s] for r, s in enumerate(sizes)]
        )

    def all_gatherv(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        if dim != 0:
            raise NotImplementedError("only dim 0 all-gatherv is supported")
        assert sizes is None or len(sizes) == self.world_size
        inputs = [input_] if isinstance(input_, torch.Tensor) else list(input_)
        if isinstance(self.dist_module, _CPUSHMDistributed):
            if sizes is None:
                sizes = [inputs[0].shape[0]] * self.world_size
            # One SHM round for all tensors (any dtype), copied straight into
            # each rank's rows: uneven sizes need no padding or concatenation.
            inputs = [t.contiguous() for t in inputs]
            outputs = [t.new_empty((sum(sizes),) + t.shape[1:]) for t in inputs]
            self.dist_module.all_gatherv(inputs, outputs, sizes)
        else:
            outputs = [self._all_gatherv_padded(t, sizes) for t in inputs]
        return outputs[0] if isinstance(input_, torch.Tensor) else outputs

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return self.reduce_scatterv(input_, dim=dim, sizes=None)

    def reduce_scatterv(
        self, input_: torch.Tensor, dim: int = -1, sizes: list[int] | None = None
    ) -> torch.Tensor:
        return self._reduce_scatterv(input_, dim, sizes)

    def reduce_scatterv_into_output(
        self,
        input_: torch.Tensor,
        output: torch.Tensor,
        dim: int = -1,
        sizes: list[int] | None = None,
    ) -> torch.Tensor:
        return self._reduce_scatterv(input_, dim, sizes, output)

    def _reduce_scatterv(
        self,
        input_: torch.Tensor,
        dim: int,
        sizes: list[int] | None,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.world_size == 1:
            if output is None:
                return input_
            return output.copy_(input_)
        if dim < 0:
            dim += input_.dim()
        input_tensor = input_ if dim == 0 else input_.movedim(0, dim)
        if sizes is None:
            assert input_tensor.shape[0] % self.world_size == 0
            sizes = [input_tensor.shape[0] // self.world_size] * self.world_size
        assert len(sizes) == self.world_size
        assert input_tensor.shape[0] == sum(sizes)
        start = sum(sizes[: self.rank_in_group])
        size = sizes[self.rank_in_group]
        result_shape = (size,) + input_tensor.shape[1:]
        if isinstance(self.dist_module, _CPUSHMDistributed) and (
            input_tensor.is_floating_point()
        ):
            input_tensor = input_tensor.contiguous()
            # Reduce straight into the caller's buffer when it is laid out
            # like this rank's slice (the MoE combine output).
            in_place = (
                output is not None
                and dim == 0
                and output.is_contiguous()
                and output.shape == result_shape
                and output.dtype == input_tensor.dtype
            )
            result = output if in_place else input_tensor.new_empty(result_shape)
            # Equal slices let the op derive the offset from the rank, which
            # keeps shape-derived ints out of compiled graphs (TP sequence
            # parallel calls this from inside the model's graph).
            out_offset = (
                None
                if all(s == sizes[0] for s in sizes)
                else start * math.prod(input_tensor.shape[1:])
            )
            self.dist_module.reduce_scatter(result, input_tensor, out_offset)
            if in_place:
                return output
        else:
            # gloo has no variable-size reduce-scatter: all-reduce a private
            # copy, then keep this rank's chunk.
            input_tensor = input_tensor.clone(memory_format=torch.contiguous_format)
            self.all_reduce(input_tensor)
            result = input_tensor[start : start + size]
        result = result.movedim(0, dim).contiguous()
        if output is None:
            return result
        return output.copy_(result)

    def send_tensor_dict(
        self,
        tensor_dict: dict[str, torch.Tensor | Any],
        dst: int,
    ) -> None:
        if not self.supports_tensor_dict:
            raise NotImplementedError(
                "CpuCommunicator does not support tensor dict fastpath with "
                "torch.distributed backend."
            )
        return self.dist_module.send_tensor_dict(tensor_dict, dst)

    def recv_tensor_dict(
        self,
        src: int,
    ) -> dict[str, torch.Tensor | Any]:
        if not self.supports_tensor_dict:
            raise NotImplementedError(
                "CpuCommunicator does not support tensor dict fastpath with "
                "torch.distributed backend."
            )
        return self.dist_module.recv_tensor_dict(src)

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """Dispatch the hidden states and router logits to the appropriate device.
        This is a no-op in the base class.
        """
        assert self.all2all_manager is not None
        return self.all2all_manager.dispatch_router_logits(
            hidden_states,
            router_logits,
            is_sequence_parallel,
            extra_tensors,
        )

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """Dispatch the hidden states and topk weights/ids to the appropriate device.
        This is a no-op in the base class.
        """
        assert self.all2all_manager is not None
        return self.all2all_manager.dispatch(
            hidden_states,
            topk_weights,
            topk_ids,
            is_sequence_parallel,
            extra_tensors=extra_tensors,
        )

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        """Combine the hidden states and router logits from the appropriate device.
        This is a no-op in the base class.
        """
        assert self.all2all_manager is not None
        return self.all2all_manager.combine(
            hidden_states,
            is_sequence_parallel,
        )

    def combine_into_output(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        is_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        assert self.all2all_manager is not None
        return self.all2all_manager.combine_into_output(
            hidden_states, output, is_sequence_parallel
        )


class _CPUSHMDistributed:
    def __init__(self, communicator: CpuCommunicator, group_name: str):
        self.communicator = communicator

        self.group_name = group_name

        self.handle = self._init_cpu_shm()

    @staticmethod
    def make_group_name(communicator: CpuCommunicator) -> str:
        instance_identifier = os.environ["VLLM_DIST_IDENT"]
        unique_name = communicator.unique_name
        instance_identifier = f"{instance_identifier}-{unique_name}"
        group_ranks = [str(rank) for rank in communicator.ranks]
        shm_group_identifier = f"[{'-'.join(group_ranks)}]"
        return f"{instance_identifier}-{shm_group_identifier}-cpushm"

    def _init_cpu_shm(self) -> int:
        thread_num_tensor = torch.tensor(
            [torch.get_num_threads()],
            dtype=torch.int64,
        )
        torch.distributed.all_reduce(
            thread_num_tensor,
            op=torch.distributed.ReduceOp.MIN,
            group=self.communicator.device_group,
        )
        thread_num = thread_num_tensor.item()

        handle = torch.ops._C.init_shm_manager(
            self.group_name,
            self.communicator.world_size,
            self.communicator.rank,
            thread_num,
        )
        torch.distributed.barrier(self.communicator.device_group)
        torch.ops._C.join_shm_manager(
            handle,
            self.group_name,
        )
        torch.distributed.barrier(self.communicator.device_group)

        return handle

    def all_reduce(
        self, input: torch.Tensor, group: ProcessGroup | None = None
    ) -> None:
        torch.ops._C.shm_allreduce(self.handle, input)

    def gather(
        self,
        input: torch.Tensor,
        gather_list: list[torch.Tensor] | None,
        dst: int = -1,
        group: ProcessGroup | None = None,
    ) -> None:
        # Note: different from the torch gather, here we use local dst rank.
        torch.ops._C.shm_gather(
            self.handle,
            input,
            gather_list,
            torch.distributed.get_group_rank(group, dst),
        )

    def all_gather_into_tensor(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        group: ProcessGroup | None = None,
    ) -> None:
        torch.ops._C.shm_all_gather(self.handle, input, output)

    def all_gatherv(
        self,
        inputs: list[torch.Tensor],
        outputs: list[torch.Tensor],
        sizes: list[int],
    ) -> None:
        torch.ops._C.shm_all_gatherv(self.handle, inputs, outputs, sizes)

    def reduce_scatter(
        self, output: torch.Tensor, input: torch.Tensor, out_offset: int | None
    ) -> None:
        # out_offset: element offset of this rank's slice in input, None for
        # equal slices ordered by rank
        torch.ops._C.shm_reduce_scatter(self.handle, input, output, out_offset)

    def send_tensor_dict(
        self,
        tensor_dict: dict[str, torch.Tensor | Any],
        dst: int,
    ) -> None:
        key_list = list(tensor_dict.keys())
        value_list = list(tensor_dict.values())
        size_list = []
        for v in value_list:
            if not isinstance(v, torch.Tensor):
                raise RuntimeError("CpuCommunicator only supports sending tensors.")
            size_list.append(v.size())
        key_size_tensor = torch.frombuffer(
            pickle.dumps([key_list, size_list]), dtype=torch.uint8
        )
        value_list.append(key_size_tensor)

        torch.ops._C.shm_send_tensor_list(self.handle, value_list, dst)

        return None

    def recv_tensor_dict(
        self,
        src: int,
    ) -> dict[str, torch.Tensor | Any]:
        tensor_list = torch.ops._C.shm_recv_tensor_list(self.handle, src)

        value_list: list[torch.Tensor] = tensor_list[:-1]
        key_size_tensor = tensor_list[-1]

        key_size = pickle.loads(key_size_tensor.numpy().tobytes())
        key_list = key_size[0]
        size_list = key_size[1]
        assert len(key_list) == len(size_list)
        assert len(key_list) == len(value_list)

        tensor_dict: dict[str, torch.Tensor] = {}
        for key, size, t in zip(key_list, size_list, value_list):
            tensor_dict[key] = t.view(size)
        return tensor_dict
