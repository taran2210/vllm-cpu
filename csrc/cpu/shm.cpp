#include "cpu/cpu_types.hpp"

#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__aarch64__) || defined(__powerpc64__)
  #include <atomic>
#endif

namespace {
#define MAX_SHM_RANK_NUM 8
#define PER_THREAD_SHM_BUFFER_BYTES (4 * 1024 * 1024)
static_assert(PER_THREAD_SHM_BUFFER_BYTES % 2 == 0);
#define PER_THREAD_SHM_BUFFER_OFFSET (PER_THREAD_SHM_BUFFER_BYTES >> 1)
#define MIN_THREAD_PROCESS_SIZE (256)
#define MAX_P2P_SEND_TENSOR_NUM 8
#define RS_BLOCK_BYTES (4 * 1024)
#define RS_BLOCKED_MIN_BYTES_X_RANKS (128LL * 1024 * 1024)
static_assert((PER_THREAD_SHM_BUFFER_BYTES >> 1) % RS_BLOCK_BYTES == 0);

template <typename scalar_t>
struct KernelVecType {
  using scalar_vec_t = void;
};

template <>
struct KernelVecType<float> {
  using scalar_vec_t = vec_op::FP32Vec16;
};

template <>
struct KernelVecType<c10::BFloat16> {
  using scalar_vec_t = vec_op::BF16Vec16;
};

template <>
struct KernelVecType<c10::Half> {
  using scalar_vec_t = vec_op::FP16Vec16;
};

struct ThreadSHMContext {
#if defined(__aarch64__) || defined(__powerpc64__)
  // memory model is weaker on AArch64, so we use atomic variables for
  // consumer (load-acquire) and producer (store-release) to make sure
  // that a stamp cannot be ready before the corresponding data is ready.
  std::atomic<char> _curr_thread_stamp[2];
  std::atomic<char> _ready_thread_stamp[2];
  static_assert(std::atomic<char>::is_always_lock_free);
#else
  volatile char _curr_thread_stamp[2];
  volatile char _ready_thread_stamp[2];
#endif  // __aarch64__
  int local_stamp_buffer_idx;
  int remote_stamp_buffer_idx;
  int thread_id;
  int thread_num;
  int rank;
  int group_size;
  size_t _spinning_count;
  int swizzled_ranks[MAX_SHM_RANK_NUM];
  void* thread_shm_ptrs[MAX_SHM_RANK_NUM];
  ThreadSHMContext* shm_contexts[MAX_SHM_RANK_NUM];
  size_t _thread_buffer_mask[2];
  char _padding2[40];

  ThreadSHMContext(const int thread_id, const int thread_num, const int rank,
                   const int group_size, void* thread_shm_ptr)
      : local_stamp_buffer_idx(0),
        remote_stamp_buffer_idx(0),
        thread_id(thread_id),
        thread_num(thread_num),
        rank(rank),
        group_size(group_size),
        _spinning_count(0) {
    static_assert(sizeof(ThreadSHMContext) % 64 == 0);
    TORCH_CHECK(group_size <= MAX_SHM_RANK_NUM);
    TORCH_CHECK((size_t)this % 64 == 0);
    TORCH_CHECK((size_t)thread_shm_ptr % 64 == 0);
#if defined(__aarch64__) || defined(__powerpc64__)
    _curr_thread_stamp[0].store(1, std::memory_order_relaxed);
    _curr_thread_stamp[1].store(1, std::memory_order_relaxed);
    _ready_thread_stamp[0].store(0, std::memory_order_relaxed);
    _ready_thread_stamp[1].store(0, std::memory_order_relaxed);
#else
    _curr_thread_stamp[0] = 1;
    _curr_thread_stamp[1] = 1;
    _ready_thread_stamp[0] = 0;
    _ready_thread_stamp[1] = 0;
#endif  // __aarch64__
    _thread_buffer_mask[0] = 0;
    _thread_buffer_mask[1] = 0;
    for (int i = 0; i < MAX_SHM_RANK_NUM; ++i) {
      shm_contexts[i] = nullptr;
      thread_shm_ptrs[i] = nullptr;
      swizzled_ranks[i] = (i + rank) % group_size;
    }
    set_context(rank, this, thread_shm_ptr);
  }

  void set_stamp_buffer_idx(int local, int remote) {
    local_stamp_buffer_idx = local;
    remote_stamp_buffer_idx = remote;
  }

  void set_context(int rank, ThreadSHMContext* ptr, void* thread_shm_ptr) {
    TORCH_CHECK(rank < MAX_SHM_RANK_NUM);
    TORCH_CHECK(ptr);
    TORCH_CHECK(thread_shm_ptr);
    TORCH_CHECK_EQ(ptr->thread_num, thread_num);
    TORCH_CHECK_EQ(ptr->thread_id, thread_id);
    shm_contexts[rank] = ptr;
    thread_shm_ptrs[rank] = thread_shm_ptr;
  }

  template <typename T>
  T* get_thread_shm_ptr(int rank) {
    return reinterpret_cast<T*>(
        reinterpret_cast<int8_t*>(thread_shm_ptrs[rank]) +
        (PER_THREAD_SHM_BUFFER_OFFSET &
         _thread_buffer_mask[local_stamp_buffer_idx]));
  }

  void next_buffer() {
    _thread_buffer_mask[local_stamp_buffer_idx] ^= 0xFFFFFFFFFFFFFFFF;
  }

  char get_curr_stamp(int idx) const {
#if defined(__aarch64__) || defined(__powerpc64__)
    return _curr_thread_stamp[idx].load(std::memory_order_acquire);
#else
    return _curr_thread_stamp[idx];
#endif  // __aarch64__
  }

  char get_ready_stamp(int idx) const {
#if defined(__aarch64__) || defined(__powerpc64__)
    return _ready_thread_stamp[idx].load(std::memory_order_acquire);
#else
    return _ready_thread_stamp[idx];
#endif  // __aarch64__
  }

  void next_stamp() {
#if defined(__aarch64__) || defined(__powerpc64__)
    _curr_thread_stamp[local_stamp_buffer_idx].fetch_add(
        1, std::memory_order_release);
#else
    _mm_mfence();
    _curr_thread_stamp[local_stamp_buffer_idx] += 1;
#endif  // __aarch64__
  }

  void commit_ready_stamp() {
#if defined(__aarch64__) || defined(__powerpc64__)
    _ready_thread_stamp[local_stamp_buffer_idx].store(
        _curr_thread_stamp[local_stamp_buffer_idx].load(
            std::memory_order_relaxed),
        std::memory_order_release);
#else
    _mm_mfence();
    _ready_thread_stamp[local_stamp_buffer_idx] =
        _curr_thread_stamp[local_stamp_buffer_idx];
#endif  // __aarch64__
  }

  int get_swizzled_rank(int idx) { return swizzled_ranks[idx]; }

  template <typename Cond>
  void wait_for_all(Cond&& cond) {
    for (int idx = 1; idx < group_size; ++idx) {
      int rank = get_swizzled_rank(idx);
      wait_for_one(rank, std::forward<Cond>(cond));
    }
  }

  template <typename Cond>
  void wait_for_one(int rank, Cond&& cond) {
    ThreadSHMContext* rank_ctx = shm_contexts[rank];
    for (;;) {
      char local_curr_stamp = get_curr_stamp(local_stamp_buffer_idx);
      char local_ready_stamp = get_ready_stamp(local_stamp_buffer_idx);
      char rank_curr_stamp = rank_ctx->get_curr_stamp(remote_stamp_buffer_idx);
      char rank_ready_stamp =
          rank_ctx->get_ready_stamp(remote_stamp_buffer_idx);
      if (cond(local_curr_stamp, local_ready_stamp, rank_curr_stamp,
               rank_ready_stamp)) {
        break;
      }
      ++_spinning_count;
#if defined(__aarch64__)
      __asm__ __volatile__("yield");
#elif defined(__powerpc64__)
      __asm__ __volatile__("or 1,1,1");
#else
      _mm_pause();
#endif  // __aarch64__
    }
  }

  static bool check_no_buffer_conflict(char local_curr_stamp,
                                       char local_ready_stamp,
                                       char rank_curr_stamp,
                                       char rank_ready_stamp) {
    char temp = rank_curr_stamp + 2;
    return local_curr_stamp != temp;
  }

  static bool check_stamp_ready(char local_curr_stamp, char local_ready_stamp,
                                char rank_curr_stamp, char rank_ready_stamp) {
    char temp = local_curr_stamp + 1;
    return (local_curr_stamp == rank_ready_stamp) || (temp == rank_ready_stamp);
  }

  std::string to_string() const {
    std::stringstream ss;
    ss << "SHMContext:";
    ss << "\nrank: " << rank;
    ss << "\ngroup_size: " << group_size;
    ss << "\nthread_num: " << thread_num;
    ss << "\nthread_id: " << thread_id;

    ss << "\nshm_ctx_stat_loop_seq: [";
    for (int i = 0; i < group_size; ++i) {
      ss << swizzled_ranks[i] << ", ";
    }
    ss << "]";

    ss << "\nshm_contexts: [";
    for (int i = 0; i < group_size; ++i) {
      if (shm_contexts[i]) {
        ss << shm_contexts[i]->rank << ", ";
      }
    }
    ss << "]";

    return ss.str();
  }
};

class SHMManager {
 public:
  explicit SHMManager(const std::string& name, const int rank,
                      const int group_size, const int thread_num)
      : _rank(rank),
        _group_size(group_size),
        _thread_num(thread_num),
        _shm_names({""}),
        _shared_mem_ptrs({nullptr}),
        _shm_ctx(nullptr) {
    _shm_names[rank] = get_shm_name(name, rank);
    _shared_mem_ptrs[rank] = init_shm(rank);
    _shm_ctx = reinterpret_cast<ThreadSHMContext*>(_shared_mem_ptrs[rank]);

    for (int i = 0; i < _thread_num; ++i) {
      ThreadSHMContext* ctx = new (_shm_ctx + i)
          ThreadSHMContext(i, _thread_num, _rank, _group_size,
                           compute_thread_shm_ptr(_shm_ctx, i));
    }
  }

  void join(const std::string& name) {
    for (int rank_idx = 0; rank_idx < _group_size; ++rank_idx) {
      if (rank_idx != _rank) {
        TORCH_CHECK(_shm_names[rank_idx].empty());
        TORCH_CHECK(_shared_mem_ptrs[rank_idx] == nullptr);
        _shm_names[rank_idx] = get_shm_name(name, rank_idx);
        _shared_mem_ptrs[rank_idx] = init_shm(rank_idx);
        ThreadSHMContext* target_ctx =
            reinterpret_cast<ThreadSHMContext*>(_shared_mem_ptrs[rank_idx]);
        for (int thread_idx = 0; thread_idx < _thread_num; ++thread_idx) {
          _shm_ctx[thread_idx].set_context(
              rank_idx, target_ctx + thread_idx,
              compute_thread_shm_ptr(target_ctx, thread_idx));
        }
      }
    }
  }

  ~SHMManager() { destroy_shm(); }

  ThreadSHMContext* get_shm_ctx() const { return _shm_ctx; }

  static std::string get_shm_name(const std::string& name, int rank) {
    return name + "_" + std::to_string(rank);
  }

  static int64_t create_singleton_instance(const std::string& name,
                                           const int group_size, const int rank,
                                           const int thread_num) {
    std::lock_guard<std::mutex> guard(SingletonInstancesLock);
    SingletonInstances.emplace_back(
        std::make_unique<SHMManager>(name, rank, group_size, thread_num));
    return static_cast<int64_t>(SingletonInstances.size() - 1);
  }

  static SHMManager* get_singleton_instance(int64_t handle) {
    return SingletonInstances[handle].get();
  }

 protected:
  static std::vector<std::unique_ptr<SHMManager>> SingletonInstances;
  static std::mutex SingletonInstancesLock;

 private:
  static size_t round_to_alignment(size_t num) {
    return ((num + 63) / 64) * 64;
  }

  int8_t* compute_thread_shm_ptr(ThreadSHMContext* ctx, int thread_id) {
    int8_t* thread_shm_ptr =
        reinterpret_cast<int8_t*>(ctx) +
        round_to_alignment(_thread_num * sizeof(ThreadSHMContext));
    return thread_shm_ptr +
           thread_id * round_to_alignment(PER_THREAD_SHM_BUFFER_BYTES);
  }

  size_t compute_shm_size() {
    const size_t rounded_rank_buffer_size =
        round_to_alignment(PER_THREAD_SHM_BUFFER_BYTES) * _thread_num;
    const size_t rounded_thread_shm_ctx_size =
        round_to_alignment(_thread_num * sizeof(ThreadSHMContext));
    const size_t shm_size =
        rounded_thread_shm_ctx_size + rounded_rank_buffer_size;
    return shm_size;
  }

  void* init_shm(int target_rank) {
    const std::string& shm_name = _shm_names[target_rank];
    const int local_rank = _rank;
    const size_t shm_size = compute_shm_size();

    int fd = -1;
    if (local_rank == target_rank) {
      fd = shm_open(shm_name.c_str(), O_CREAT | O_EXCL | O_RDWR,
                    S_IRUSR | S_IWUSR);

      if (fd == -1)
        TORCH_CHECK(false, "create shm in SHMManager failed. errno: " +
                               std::to_string(errno));

      if (ftruncate(fd, shm_size) == -1)
        TORCH_CHECK(false, "ftruncate in SHMManager failed. errno: " +
                               std::to_string(errno));
    } else {
      fd = shm_open(shm_name.c_str(), O_RDWR, S_IRUSR | S_IWUSR);

      if (fd == -1)
        TORCH_CHECK(false, "open shm in SHMManager failed. errno: " +
                               std::to_string(errno));
    }

    void* shm_ptr = mmap(nullptr, shm_size, PROT_READ | PROT_WRITE,
                         MAP_SHARED | MAP_POPULATE, fd, 0);

    if (shm_ptr == MAP_FAILED) {
      TORCH_CHECK(false,
                  "mmap in SHMManager failed. errno: " + std::to_string(errno));
    }

    if (close(fd) != 0) {
      TORCH_CHECK(
          false, "close in SHMManager failed. errno: " + std::to_string(errno));
    }

    TORCH_CHECK((size_t)shm_ptr % 64 == 0);

    return shm_ptr;
  }

  void destroy_shm() {
    std::stringstream ss;
    ss << "local rank " << _rank << ": [";
    for (int thread_id = 0; thread_id < _thread_num; ++thread_id) {
      ss << _shm_ctx[thread_id]._spinning_count << ", ";
    }
    ss << "]\n";

    for (int i = 0; i < MAX_SHM_RANK_NUM; ++i) {
      if (_shared_mem_ptrs[i] != nullptr) {
        munmap(_shared_mem_ptrs[i], compute_shm_size());
      }

      if (!_shm_names[i].empty()) {
        shm_unlink(_shm_names[i].c_str());
      }
    }
  }

  int _rank;
  int _group_size;
  int _thread_num;
  std::array<std::string, MAX_SHM_RANK_NUM> _shm_names;
  std::array<void*, MAX_SHM_RANK_NUM> _shared_mem_ptrs;
  ThreadSHMContext* _shm_ctx;
};

namespace shm_cc_ops {
template <typename scalar_t, typename F>
void shm_cc_loop(ThreadSHMContext* ctx, int64_t elem_num, F&& inner_func) {
  int thread_num = ctx->thread_num;
  int64_t total_bytes = elem_num * sizeof(scalar_t);
  int64_t total_units_num =
      (total_bytes + MIN_THREAD_PROCESS_SIZE - 1) / MIN_THREAD_PROCESS_SIZE;
  int64_t per_thread_units_num =
      (total_units_num + thread_num - 1) / thread_num;
  int64_t per_unit_elem_num = MIN_THREAD_PROCESS_SIZE / sizeof(scalar_t);
  int64_t max_per_thread_iteration_elem_num =
      (PER_THREAD_SHM_BUFFER_BYTES >> 1) /
      sizeof(scalar_t);  // Note: double buffer
  int64_t per_thread_elem_num = per_unit_elem_num * per_thread_units_num;

#pragma omp parallel for schedule(static, 1)
  for (int i = 0; i < thread_num; ++i) {
    int64_t offset = i * per_thread_elem_num;
    int64_t end = std::min(elem_num, offset + per_thread_elem_num);
    int64_t curr_elem_num =
        std::min(max_per_thread_iteration_elem_num, end - offset);
    ThreadSHMContext* thread_ctx = ctx + i;
    bool fast_mode = ((end - offset) <= max_per_thread_iteration_elem_num);

    while (curr_elem_num > 0) {
      inner_func(thread_ctx, offset, curr_elem_num, fast_mode);

      thread_ctx->next_stamp();
      thread_ctx->next_buffer();
      offset += max_per_thread_iteration_elem_num;
      curr_elem_num = std::min(max_per_thread_iteration_elem_num, end - offset);
    }
  }
}

void reset_threads_stamp_buffer_idx(ThreadSHMContext* ctx, int local,
                                    int remote) {
  int thread_num = ctx->thread_num;
  for (int i = 0; i < thread_num; ++i) {
    ThreadSHMContext* thread_ctx = ctx + i;
    thread_ctx->set_stamp_buffer_idx(local, remote);
  }
}
};  // namespace shm_cc_ops

namespace shm_cc_ops {

void memcpy_from_shm(void* dst, void* src, const int64_t bytes) {
  const int64_t aligned_bytes = ((bytes >> 6) << 6);  // 64 bytes aligned
  int64_t i = 0;
#pragma GCC unroll 4
  for (; i < aligned_bytes; i += 64) {
    vec_op::INT8Vec64 data(
        true, (int8_t*)src + i);  // stream loading shm to avoid caching
    data.save((int8_t*)dst + i);
  }
  if (aligned_bytes < bytes) {
    vec_op::INT8Vec64 data(true, (int8_t*)src + aligned_bytes);
    data.save((int8_t*)dst + aligned_bytes, bytes - aligned_bytes);
  }
}

void memcpy_to_shm(void* dst, void* src, const int64_t bytes) {
#pragma GCC unroll 4
  for (int64_t i = 0; i < bytes; i += 64) {
    vec_op::INT8Vec64 data((int8_t*)src + i);
    data.nt_save((int8_t*)dst + i);
  }
}

void memcpy(void* dst, void* src, const int64_t bytes) {
  const int64_t aligned_bytes = ((bytes >> 6) << 6);  // 64 bytes aligned
  int64_t i = 0;
#pragma GCC unroll 4
  for (; i < aligned_bytes; i += 64) {
    vec_op::INT8Vec64 data((int8_t*)src + i);
    data.save((int8_t*)dst + i);
  }
  if (aligned_bytes < bytes) {
    vec_op::INT8Vec64 data((int8_t*)src + aligned_bytes);
    data.save((int8_t*)dst + aligned_bytes, bytes - aligned_bytes);
  }
}

template <typename scalar_t, int RANKS>
void all_reduce_sum_impl(ThreadSHMContext* ctx, scalar_t* data,
                         size_t elem_num) {
  CPU_KERNEL_GUARD_IN(all_reduce_sum_impl)
  using vec_t = typename KernelVecType<scalar_t>::scalar_vec_t;
  constexpr int64_t vec_elem_num = vec_t::get_elem_num();
  const int worldsize = ctx->group_size;

  shm_cc_ops::shm_cc_loop<scalar_t>(
      ctx, elem_num,
      [&](ThreadSHMContext* thread_ctx, int64_t data_offset,
          int64_t data_elem_num, bool fast_mode) {
        int rank = thread_ctx->rank;
        scalar_t* thread_shm_ptr =
            thread_ctx->get_thread_shm_ptr<scalar_t>(rank);
        scalar_t* thread_data_ptr = data + data_offset;
        int64_t thread_data_elem_num = data_elem_num * sizeof(scalar_t);

        scalar_t* remote_data_ptrs[RANKS - 1];
        vec_op::unroll_loop<int, RANKS - 1>([&](int idx) {
          remote_data_ptrs[idx] = thread_ctx->get_thread_shm_ptr<scalar_t>(
              thread_ctx->get_swizzled_rank(idx + 1));
        });

        if (!fast_mode) {
          thread_ctx->wait_for_all(ThreadSHMContext::check_no_buffer_conflict);
        }

        shm_cc_ops::memcpy_to_shm(thread_shm_ptr, thread_data_ptr,
                                  thread_data_elem_num);
        thread_ctx->commit_ready_stamp();
        int64_t aligned_data_elem_num =
            (data_elem_num / vec_elem_num) * vec_elem_num;
        int64_t i = 0;
        thread_ctx->wait_for_all(ThreadSHMContext::check_stamp_ready);
#pragma GCC unroll 4
        for (; i < aligned_data_elem_num; i += vec_elem_num) {
          vec_t local_data(thread_data_ptr + i);  // load from cache
          vec_op::FP32Vec16 local_data_fp32(local_data);
          vec_op::unroll_loop<int, RANKS - 1>([&](int idx) {
            vec_t remote_data(
                true, remote_data_ptrs[idx] + i);  // stream load from shm
            vec_op::FP32Vec16 remote_data_fp32(remote_data);
            local_data_fp32 = local_data_fp32 + remote_data_fp32;  // sum reduce
          });
          vec_t reduced_data(local_data_fp32);
          reduced_data.save(thread_data_ptr + i);
        }

        if (i < data_elem_num) {
          vec_t local_data(thread_data_ptr + i);  // load from cache
          vec_op::FP32Vec16 local_data_fp32(local_data);
          vec_op::unroll_loop<int, RANKS - 1>([&](int idx) {
            vec_t remote_data(
                true, remote_data_ptrs[idx] + i);  // stream load from shm
            vec_op::FP32Vec16 remote_data_fp32(remote_data);
            local_data_fp32 = local_data_fp32 + remote_data_fp32;  // sum reduce
          });
          vec_t reduced_data(local_data_fp32);
          reduced_data.save(thread_data_ptr + i,
                            data_elem_num - aligned_data_elem_num);
        }
      });

  return;
}

// Same protocol as all_reduce_sum_impl, but each rank only reduces its own
// slice [out_begin, out_end) of `data` and writes it to `output`. Each thread
// takes one contiguous range, matching how the producing kernel usually split
// `data`, so threads read what is still in their own cache. Only the threads
// whose range overlaps this rank's slice reduce, which costs little with two
// ranks. Ranges entirely inside this rank's slice are never read remotely and
// are not copied to SHM.
template <typename scalar_t, int RANKS>
void reduce_scatter_sum_contiguous_impl(ThreadSHMContext* ctx, scalar_t* data,
                                        size_t elem_num, scalar_t* output,
                                        int64_t out_begin, int64_t out_end) {
  CPU_KERNEL_GUARD_IN(reduce_scatter_sum_contiguous_impl)
  using vec_t = typename KernelVecType<scalar_t>::scalar_vec_t;
  constexpr int64_t vec_elem_num = vec_t::get_elem_num();

  shm_cc_ops::shm_cc_loop<scalar_t>(
      ctx, elem_num,
      [&](ThreadSHMContext* thread_ctx, int64_t data_offset,
          int64_t data_elem_num, bool fast_mode) {
        int rank = thread_ctx->rank;
        scalar_t* thread_shm_ptr =
            thread_ctx->get_thread_shm_ptr<scalar_t>(rank);
        scalar_t* thread_data_ptr = data + data_offset;

        scalar_t* remote_data_ptrs[RANKS - 1];
        vec_op::unroll_loop<int, RANKS - 1>([&](int idx) {
          remote_data_ptrs[idx] = thread_ctx->get_thread_shm_ptr<scalar_t>(
              thread_ctx->get_swizzled_rank(idx + 1));
        });

        if (!fast_mode) {
          thread_ctx->wait_for_all(ThreadSHMContext::check_no_buffer_conflict);
        }

        if (data_offset < out_begin || data_offset + data_elem_num > out_end) {
          shm_cc_ops::memcpy_to_shm(thread_shm_ptr, thread_data_ptr,
                                    data_elem_num * sizeof(scalar_t));
        }
        thread_ctx->commit_ready_stamp();

        // part of this chunk that belongs to this rank's output, may start
        // at any element so use unaligned loads
        const int64_t begin = std::max(data_offset, out_begin) - data_offset;
        const int64_t end =
            std::min(data_offset + data_elem_num, out_end) - data_offset;
        thread_ctx->wait_for_all(ThreadSHMContext::check_stamp_ready);
        if (begin >= end) {
          return;
        }

        scalar_t* out_ptr = output + (data_offset + begin - out_begin);
        int64_t i = begin;
        for (; i + vec_elem_num <= end; i += vec_elem_num) {
          vec_t local_data(thread_data_ptr + i);
          vec_op::FP32Vec16 acc(local_data);
          vec_op::unroll_loop<int, RANKS - 1>([&](int idx) {
            vec_t remote_data(remote_data_ptrs[idx] + i);
            acc = acc + vec_op::FP32Vec16(remote_data);
          });
          vec_t(acc).save(out_ptr + (i - begin));
        }
        for (; i < end; ++i) {
          float acc = static_cast<float>(thread_data_ptr[i]);
          vec_op::unroll_loop<int, RANKS - 1>([&](int idx) {
            acc += static_cast<float>(remote_data_ptrs[idx][i]);
          });
          out_ptr[i - begin] = static_cast<scalar_t>(acc);
        }
      });
}

// Same as reduce_scatter_sum_contiguous_impl, but elements are dealt to
// threads in RS_BLOCK_BYTES blocks round-robin, so every rank's slice is
// spread over all threads instead of the 1/RANKS of threads whose range
// overlaps it. All ranks see the same elem_num, so they agree on the block
// layout without knowing each other's slices.
template <typename scalar_t, int RANKS>
void reduce_scatter_sum_blocked_impl(ThreadSHMContext* ctx, scalar_t* data,
                                     size_t elem_num, scalar_t* output,
                                     int64_t out_begin, int64_t out_end) {
  CPU_KERNEL_GUARD_IN(reduce_scatter_sum_blocked_impl)
  using vec_t = typename KernelVecType<scalar_t>::scalar_vec_t;
  constexpr int64_t vec_elem_num = vec_t::get_elem_num();
  constexpr int64_t block_elem_num = RS_BLOCK_BYTES / sizeof(scalar_t);
  constexpr int64_t round_block_capacity =
      (PER_THREAD_SHM_BUFFER_BYTES >> 1) / RS_BLOCK_BYTES;
  static_assert(block_elem_num % vec_elem_num == 0);
  const int thread_num = ctx->thread_num;
  const int64_t total = elem_num;
  const int64_t block_num = (total + block_elem_num - 1) / block_elem_num;

#pragma omp parallel for schedule(static, 1)
  for (int t = 0; t < thread_num; ++t) {
    ThreadSHMContext* thread_ctx = ctx + t;
    // this thread owns global blocks t, t + thread_num, t + 2 * thread_num...
    const int64_t thread_block_num =
        block_num > t ? (block_num - t + thread_num - 1) / thread_num : 0;
    const bool fast_mode = thread_block_num <= round_block_capacity;

    for (int64_t round_begin = 0; round_begin < thread_block_num;
         round_begin += round_block_capacity) {
      const int64_t round_block_num =
          std::min(round_block_capacity, thread_block_num - round_begin);
      scalar_t* local_shm =
          thread_ctx->get_thread_shm_ptr<scalar_t>(thread_ctx->rank);
      scalar_t* remote_shm[RANKS - 1];
      vec_op::unroll_loop<int, RANKS - 1>([&](int idx) {
        remote_shm[idx] = thread_ctx->get_thread_shm_ptr<scalar_t>(
            thread_ctx->get_swizzled_rank(idx + 1));
      });

      if (!fast_mode) {
        thread_ctx->wait_for_all(ThreadSHMContext::check_no_buffer_conflict);
      }

      for (int64_t b = 0; b < round_block_num; ++b) {
        const int64_t begin =
            ((round_begin + b) * thread_num + t) * block_elem_num;
        const int64_t end = std::min(begin + block_elem_num, total);
        if (begin >= out_begin && end <= out_end) {
          continue;
        }
        const int64_t bytes = (end - begin) * sizeof(scalar_t);
        const int64_t aligned_bytes = (bytes >> 6) << 6;
        scalar_t* dst = local_shm + b * block_elem_num;
        shm_cc_ops::memcpy_to_shm(dst, data + begin, aligned_bytes);
        if (aligned_bytes < bytes) {
          std::memcpy(reinterpret_cast<int8_t*>(dst) + aligned_bytes,
                      reinterpret_cast<int8_t*>(data + begin) + aligned_bytes,
                      bytes - aligned_bytes);
        }
      }
      thread_ctx->commit_ready_stamp();
      thread_ctx->wait_for_all(ThreadSHMContext::check_stamp_ready);

      for (int64_t b = 0; b < round_block_num; ++b) {
        const int64_t begin =
            ((round_begin + b) * thread_num + t) * block_elem_num;
        const int64_t lo = std::max(begin, out_begin);
        const int64_t hi = std::min({begin + block_elem_num, total, out_end});
        if (lo >= hi) {
          continue;
        }
        // the slice may start at any element, so use unaligned loads
        const int64_t shm_off = b * block_elem_num + (lo - begin);
        const scalar_t* local_ptr = data + lo;
        scalar_t* out_ptr = output + (lo - out_begin);
        const int64_t len = hi - lo;
        int64_t i = 0;
        for (; i + vec_elem_num <= len; i += vec_elem_num) {
          vec_t local_data(local_ptr + i);
          vec_op::FP32Vec16 acc(local_data);
          vec_op::unroll_loop<int, RANKS - 1>([&](int idx) {
            vec_t remote_data(remote_shm[idx] + shm_off + i);
            acc = acc + vec_op::FP32Vec16(remote_data);
          });
          vec_t(acc).save(out_ptr + i);
        }
        for (; i < len; ++i) {
          float acc = static_cast<float>(local_ptr[i]);
          vec_op::unroll_loop<int, RANKS - 1>([&](int idx) {
            acc += static_cast<float>(remote_shm[idx][shm_off + i]);
          });
          out_ptr[i] = static_cast<scalar_t>(acc);
        }
      }

      thread_ctx->next_stamp();
      thread_ctx->next_buffer();
    }
  }
}

// Round-robin blocks balance the reduction over all threads but give up the
// producer's cache locality, which only pays off for large inputs spread over
// many ranks (measured crossover: ~21 MB at 6 ranks, ~32 MB at 4 ranks).
template <typename scalar_t, int RANKS>
void reduce_scatter_sum_impl(ThreadSHMContext* ctx, scalar_t* data,
                             size_t elem_num, scalar_t* output,
                             int64_t out_begin, int64_t out_end) {
  if (RANKS >= 3 && static_cast<int64_t>(elem_num * sizeof(scalar_t)) * RANKS >=
                        RS_BLOCKED_MIN_BYTES_X_RANKS) {
    reduce_scatter_sum_blocked_impl<scalar_t, RANKS>(
        ctx, data, elem_num, output, out_begin, out_end);
  } else {
    reduce_scatter_sum_contiguous_impl<scalar_t, RANKS>(
        ctx, data, elem_num, output, out_begin, out_end);
  }
}
};  // namespace shm_cc_ops

std::vector<std::unique_ptr<SHMManager>> SHMManager::SingletonInstances = {};
std::mutex SHMManager::SingletonInstancesLock = {};

template <typename scalar_t>
void shm_allreduce_sum(ThreadSHMContext* ctx, scalar_t* data, size_t elem_num) {
  switch (ctx->group_size) {
    case 2:
      shm_cc_ops::all_reduce_sum_impl<scalar_t, 2>(ctx, data, elem_num);
      break;
    case 3:
      shm_cc_ops::all_reduce_sum_impl<scalar_t, 3>(ctx, data, elem_num);
      break;
    case 4:
      shm_cc_ops::all_reduce_sum_impl<scalar_t, 4>(ctx, data, elem_num);
      break;
    case 5:
      shm_cc_ops::all_reduce_sum_impl<scalar_t, 5>(ctx, data, elem_num);
      break;
    case 6:
      shm_cc_ops::all_reduce_sum_impl<scalar_t, 6>(ctx, data, elem_num);
      break;
    case 7:
      shm_cc_ops::all_reduce_sum_impl<scalar_t, 7>(ctx, data, elem_num);
      break;
    case 8:
      shm_cc_ops::all_reduce_sum_impl<scalar_t, 8>(ctx, data, elem_num);
      break;
    default:
      TORCH_CHECK(false,
                  "Invalid world size: " + std::to_string(ctx->group_size));
  }
}

template <typename scalar_t>
void shm_reduce_scatter_sum(ThreadSHMContext* ctx, scalar_t* data,
                            size_t elem_num, scalar_t* output,
                            int64_t out_begin, int64_t out_end) {
  switch (ctx->group_size) {
    case 2:
      shm_cc_ops::reduce_scatter_sum_impl<scalar_t, 2>(
          ctx, data, elem_num, output, out_begin, out_end);
      break;
    case 3:
      shm_cc_ops::reduce_scatter_sum_impl<scalar_t, 3>(
          ctx, data, elem_num, output, out_begin, out_end);
      break;
    case 4:
      shm_cc_ops::reduce_scatter_sum_impl<scalar_t, 4>(
          ctx, data, elem_num, output, out_begin, out_end);
      break;
    case 5:
      shm_cc_ops::reduce_scatter_sum_impl<scalar_t, 5>(
          ctx, data, elem_num, output, out_begin, out_end);
      break;
    case 6:
      shm_cc_ops::reduce_scatter_sum_impl<scalar_t, 6>(
          ctx, data, elem_num, output, out_begin, out_end);
      break;
    case 7:
      shm_cc_ops::reduce_scatter_sum_impl<scalar_t, 7>(
          ctx, data, elem_num, output, out_begin, out_end);
      break;
    case 8:
      shm_cc_ops::reduce_scatter_sum_impl<scalar_t, 8>(
          ctx, data, elem_num, output, out_begin, out_end);
      break;
    default:
      TORCH_CHECK(false,
                  "Invalid world size: " + std::to_string(ctx->group_size));
  }
}

template <typename scalar_t>
void shm_gather_impl(ThreadSHMContext* ctx, scalar_t* data, size_t elem_num,
                     scalar_t** outputs, const int dst) {
  CPU_KERNEL_GUARD_IN(shm_gather_impl)
  const int worldsize = ctx->group_size;
  TORCH_CHECK_LT(dst, worldsize);
  shm_cc_ops::shm_cc_loop<scalar_t>(
      ctx, elem_num,
      [&](ThreadSHMContext* thread_ctx, int64_t data_offset,
          int64_t data_elem_num, bool fast_mode) {
        int rank = thread_ctx->rank;
        scalar_t* thread_shm_ptr =
            thread_ctx->get_thread_shm_ptr<scalar_t>(rank);

        if (!fast_mode) {
          thread_ctx->wait_for_all(ThreadSHMContext::check_no_buffer_conflict);
        }

        shm_cc_ops::memcpy(thread_shm_ptr, data + data_offset,
                           data_elem_num * sizeof(scalar_t));
        thread_ctx->commit_ready_stamp();
        if (rank == dst) {
          shm_cc_ops::memcpy(outputs[rank] + data_offset, data + data_offset,
                             data_elem_num * sizeof(scalar_t));
          for (int i = 1; i < worldsize; ++i) {
            int src_rank = thread_ctx->get_swizzled_rank(i);
            scalar_t* src_ptr =
                thread_ctx->get_thread_shm_ptr<scalar_t>(src_rank);  // shm
            scalar_t* dst_ptr = outputs[src_rank] + data_offset;
            thread_ctx->wait_for_one(src_rank,
                                     ThreadSHMContext::check_stamp_ready);
            shm_cc_ops::memcpy(dst_ptr, src_ptr,
                               data_elem_num * sizeof(scalar_t));
          }
        }
      });

  return;
}

// Variable-size all-gather of several tensors along dim 0 in one SHM round.
// Each rank streams its inputs back to back ([t0 rows | t1 rows | ...]) through
// its SHM buffer and readers copy every rank's stream straight into that rank's
// rows of each output, so uneven sizes need no padding or concatenation. The
// loop runs over the longest stream so that all ranks take the same stamp steps.
void shm_all_gatherv_impl(ThreadSHMContext* ctx,
                          const std::vector<torch::Tensor>& inputs,
                          const std::vector<torch::Tensor>& outputs,
                          const std::vector<int64_t>& sizes) {
  CPU_KERNEL_GUARD_IN(shm_all_gatherv_impl)
  const int world_size = ctx->group_size;
  const int64_t tensor_num = inputs.size();
  TORCH_CHECK_EQ(sizes.size(), world_size);
  TORCH_CHECK_EQ(outputs.size(), tensor_num);
  TORCH_CHECK_LE(tensor_num, MAX_P2P_SEND_TENSOR_NUM);

  int64_t row_offsets[MAX_SHM_RANK_NUM + 1] = {0};
  int64_t max_rows = 0;
  for (int r = 0; r < world_size; ++r) {
    row_offsets[r + 1] = row_offsets[r] + sizes[r];
    max_rows = std::max(max_rows, sizes[r]);
  }
  const int64_t total_rows = row_offsets[world_size];
  if (total_rows == 0) {
    return;
  }

  int64_t row_bytes[MAX_P2P_SEND_TENSOR_NUM];
  const int8_t* in_ptrs[MAX_P2P_SEND_TENSOR_NUM];
  int8_t* out_ptrs[MAX_P2P_SEND_TENSOR_NUM];
  int64_t stream_row_bytes = 0;
  for (int64_t j = 0; j < tensor_num; ++j) {
    const torch::Tensor& in = inputs[j];
    const torch::Tensor& out = outputs[j];
    TORCH_CHECK(in.is_contiguous() && out.is_contiguous());
    TORCH_CHECK_EQ(in.scalar_type(), out.scalar_type());
    TORCH_CHECK_EQ(out.nbytes() % total_rows, 0);
    row_bytes[j] = out.nbytes() / total_rows;
    TORCH_CHECK_EQ(in.nbytes(), sizes[ctx->rank] * row_bytes[j]);
    in_ptrs[j] = reinterpret_cast<const int8_t*>(in.data_ptr());
    out_ptrs[j] = reinterpret_cast<int8_t*>(out.data_ptr());
    stream_row_bytes += row_bytes[j];
  }

  // Calls fn(tensor, offset in that tensor's rows of rank r, offset in the
  // chunk, bytes) for each piece of rank r's stream within [begin, begin+len).
  auto for_each_piece = [&](int r, int64_t begin, int64_t len, auto&& fn) {
    int64_t seg_begin = 0;
    for (int64_t j = 0; j < tensor_num; ++j) {
      const int64_t seg_end = seg_begin + sizes[r] * row_bytes[j];
      const int64_t lo = std::max(begin, seg_begin);
      const int64_t hi = std::min(begin + len, seg_end);
      if (lo < hi) {
        fn(j, lo - seg_begin, lo - begin, hi - lo);
      }
      seg_begin = seg_end;
    }
  };

  // std::memcpy: pieces start at arbitrary byte offsets, and the tail load of
  // shm_cc_ops::memcpy reads a full vector past the end of the source.
  shm_cc_ops::shm_cc_loop<int8_t>(
      ctx, max_rows * stream_row_bytes,
      [&](ThreadSHMContext* thread_ctx, int64_t data_offset,
          int64_t data_elem_num, bool fast_mode) {
        const int rank = thread_ctx->rank;
        int8_t* local_shm = thread_ctx->get_thread_shm_ptr<int8_t>(rank);

        if (!fast_mode) {
          thread_ctx->wait_for_all(ThreadSHMContext::check_no_buffer_conflict);
        }

        for_each_piece(rank, data_offset, data_elem_num,
                       [&](int64_t j, int64_t seg_off, int64_t chunk_off,
                           int64_t bytes) {
                         const int8_t* src = in_ptrs[j] + seg_off;
                         std::memcpy(local_shm + chunk_off, src, bytes);
                         std::memcpy(out_ptrs[j] + row_offsets[rank] * row_bytes[j] +
                                         seg_off,
                                     src, bytes);
                       });
        thread_ctx->commit_ready_stamp();

        for (int i = 1; i < world_size; ++i) {
          const int src_rank = thread_ctx->get_swizzled_rank(i);
          thread_ctx->wait_for_one(src_rank,
                                   ThreadSHMContext::check_stamp_ready);
          const int8_t* remote_shm =
              thread_ctx->get_thread_shm_ptr<int8_t>(src_rank);
          for_each_piece(
              src_rank, data_offset, data_elem_num,
              [&](int64_t j, int64_t seg_off, int64_t chunk_off, int64_t bytes) {
                std::memcpy(out_ptrs[j] + row_offsets[src_rank] * row_bytes[j] +
                                seg_off,
                            remote_shm + chunk_off, bytes);
              });
        }
      });
}

struct MemPiece {
  void* ptr;
  int64_t size;

  template <typename T>
  T* data_ptr() {
    return reinterpret_cast<T*>(ptr);
  }
};

struct TensorListMeta {
  int64_t tensor_bytes[MAX_P2P_SEND_TENSOR_NUM];
  torch::ScalarType tensor_types[MAX_P2P_SEND_TENSOR_NUM];
  int64_t tensor_num;
  int64_t total_bytes;

  TensorListMeta() : tensor_num(0), total_bytes(0) {
    static_assert(sizeof(TensorListMeta) % 64 == 0);
    static_assert(sizeof(TensorListMeta) <
                  MIN_THREAD_PROCESS_SIZE);  // To ensure the metadata always
                                             // hold by the thread 0
    for (int i = 0; i < MAX_P2P_SEND_TENSOR_NUM; ++i) {
      tensor_bytes[i] = 0;
      tensor_ptrs[i] = nullptr;
      tensor_types[i] = torch::ScalarType::Undefined;
    }
  }

  // For send and recv
  void bind_tensor_list(std::vector<torch::Tensor>& tensor_list) {
    TORCH_CHECK(tensor_types[0] == torch::ScalarType::Undefined,
                "Re-bind TensorListMeta is not allowed.")
    TORCH_CHECK_LE(tensor_list.size(), MAX_P2P_SEND_TENSOR_NUM);
    tensor_num = tensor_list.size();
    int64_t bytes_sum = 0;
    for (int i = 0; i < tensor_list.size(); ++i) {
      torch::Tensor& t = tensor_list[i];
      TORCH_CHECK(t.is_contiguous());
      tensor_bytes[i] = t.nbytes();
      tensor_types[i] = t.scalar_type();
      tensor_ptrs[i] = t.data_ptr();
      bytes_sum += t.nbytes();
    }
    total_bytes = bytes_sum;
  }

  // For recv
  std::vector<torch::Tensor> generate_tensor_list() {
    std::vector<torch::Tensor> tensor_list;
    tensor_list.reserve(tensor_num);

    for (int i = 0; i < tensor_num; ++i) {
      int64_t bytes = tensor_bytes[i];
      auto type = tensor_types[i];
      int64_t elem_bytes = torch::elementSize(type);

      TORCH_CHECK_EQ(bytes % elem_bytes, 0);
      int64_t elem_num = bytes / elem_bytes;
      auto options = torch::TensorOptions().dtype(type).device(torch::kCPU);
      tensor_list.emplace_back(torch::empty({elem_num}, options));
    }
    return tensor_list;
  }

  MemPiece get_data(int64_t offset) {
    for (int i = 0; i < tensor_num; ++i) {
      if (offset < tensor_bytes[i]) {
        return {reinterpret_cast<int8_t*>(tensor_ptrs[i]) + offset,
                tensor_bytes[i] - offset};
      }
      offset -= tensor_bytes[i];
    }
    return {nullptr, 0};
  }

 private:
  void* tensor_ptrs[MAX_P2P_SEND_TENSOR_NUM];
  int8_t _padding[40];
};

void shm_send_tensor_list_impl(ThreadSHMContext* ctx, int64_t dst,
                               const std::vector<torch::Tensor>& tensor_list) {
  CPU_KERNEL_GUARD_IN(shm_send_tensor_list_impl)
  std::vector<torch::Tensor> tensor_list_with_metadata;
  tensor_list_with_metadata.reserve(1 + tensor_list.size());

  auto options = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCPU);
  tensor_list_with_metadata.emplace_back(
      torch::empty({sizeof(TensorListMeta)}, options));
  tensor_list_with_metadata.insert(tensor_list_with_metadata.end(),
                                   tensor_list.begin(), tensor_list.end());

  torch::Tensor& metadata_tensor = tensor_list_with_metadata[0];
  TORCH_CHECK_EQ(metadata_tensor.nbytes(), sizeof(TensorListMeta));

  TensorListMeta* metadata = new (metadata_tensor.data_ptr()) TensorListMeta();
  metadata->bind_tensor_list(tensor_list_with_metadata);

  shm_cc_ops::reset_threads_stamp_buffer_idx(ctx, 0, 1);
  shm_cc_ops::shm_cc_loop<int8_t>(
      ctx, metadata->total_bytes,
      [&](ThreadSHMContext* thread_ctx, int64_t data_offset,
          int64_t data_elem_num, bool fast_mode) {
        int rank = thread_ctx->rank;
        int64_t curr_shm_offset = 0;
        thread_ctx->wait_for_one(dst,
                                 ThreadSHMContext::check_no_buffer_conflict);
        while (curr_shm_offset < data_elem_num) {
          MemPiece frag = metadata->get_data(data_offset + curr_shm_offset);
          frag.size = std::min(frag.size, data_elem_num - curr_shm_offset);
          shm_cc_ops::memcpy(
              thread_ctx->get_thread_shm_ptr<int8_t>(rank) + curr_shm_offset,
              frag.ptr, frag.size);
          curr_shm_offset += frag.size;
        }
        thread_ctx->commit_ready_stamp();
      });
}

std::vector<torch::Tensor> shm_recv_tensor_list_impl(ThreadSHMContext* ctx,
                                                     int64_t src) {
  CPU_KERNEL_GUARD_IN(shm_recv_tensor_list_impl)
  auto options = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCPU);
  torch::Tensor metadata_tensor =
      torch::empty({sizeof(TensorListMeta)}, options);

  shm_cc_ops::reset_threads_stamp_buffer_idx(ctx, 1, 0);
  ctx->wait_for_one(src, ThreadSHMContext::check_stamp_ready);
  shm_cc_ops::memcpy(metadata_tensor.data_ptr(),
                     ctx->get_thread_shm_ptr<void>(src),
                     sizeof(TensorListMeta));
  TensorListMeta* src_metadata =
      reinterpret_cast<TensorListMeta*>(metadata_tensor.data_ptr());
  std::vector<torch::Tensor> tensor_list_with_metadata =
      src_metadata->generate_tensor_list();

  TensorListMeta metadata;
  metadata.bind_tensor_list(tensor_list_with_metadata);
  TORCH_CHECK_EQ(metadata.tensor_num, src_metadata->tensor_num);
  TORCH_CHECK_EQ(metadata.total_bytes, src_metadata->total_bytes);

  shm_cc_ops::shm_cc_loop<int8_t>(
      ctx, metadata.total_bytes,
      [&](ThreadSHMContext* thread_ctx, int64_t data_offset,
          int64_t data_elem_num, bool fast_mode) {
        thread_ctx->wait_for_one(src, ThreadSHMContext::check_stamp_ready);
        int64_t curr_shm_offset = 0;
        while (curr_shm_offset < data_elem_num) {
          MemPiece frag = metadata.get_data(data_offset + curr_shm_offset);
          frag.size = std::min(frag.size, data_elem_num - curr_shm_offset);
          shm_cc_ops::memcpy(
              frag.ptr,
              thread_ctx->get_thread_shm_ptr<int8_t>(src) + curr_shm_offset,
              frag.size);
          curr_shm_offset += frag.size;
        }
      });

  std::vector<torch::Tensor> tensor_list;
  tensor_list.reserve(metadata.tensor_num - 1);
  tensor_list.insert(tensor_list.begin(), tensor_list_with_metadata.begin() + 1,
                     tensor_list_with_metadata.end());

  return tensor_list;
}
}  // namespace

void shm_gather(int64_t handle, torch::Tensor& data,
                const std::optional<std::vector<torch::Tensor>>& outputs,
                int64_t dst) {
  TORCH_CHECK(data.is_contiguous())
  VLLM_DISPATCH_FLOATING_TYPES(data.scalar_type(), "shm_gather_impl", [&] {
    CPU_KERNEL_GUARD_IN(shm_gather_impl)

    if (outputs.has_value()) {
      TORCH_CHECK_LE(outputs->size(), MAX_SHM_RANK_NUM);
      scalar_t* output_ptrs[MAX_SHM_RANK_NUM] = {nullptr};
      for (int i = 0; i < outputs->size(); ++i) {
        output_ptrs[i] = outputs->at(i).data_ptr<scalar_t>();
      }
      shm_gather_impl(SHMManager::get_singleton_instance(handle)->get_shm_ctx(),
                      data.data_ptr<scalar_t>(), data.numel(), output_ptrs,
                      dst);
    } else {
      shm_gather_impl(SHMManager::get_singleton_instance(handle)->get_shm_ctx(),
                      data.data_ptr<scalar_t>(), data.numel(), (scalar_t**)(0),
                      dst);
    }

    CPU_KERNEL_GUARD_OUT(shm_gather_impl)
  });
}

void shm_all_gather(int64_t handle, const torch::Tensor& data,
                    torch::Tensor& output) {
  TORCH_CHECK(data.is_contiguous())
  TORCH_CHECK(output.is_contiguous())

  const int64_t input_elem_num = data.numel();
  const int64_t output_elem_num = output.numel();
  TORCH_CHECK_EQ(output_elem_num % input_elem_num, 0);
  const int world_size = output_elem_num / input_elem_num;

  VLLM_DISPATCH_FLOATING_TYPES(data.scalar_type(), "shm_all_gather_impl", [&] {
    CPU_KERNEL_GUARD_IN(shm_all_gather_impl)
    auto ctx = SHMManager::get_singleton_instance(handle)->get_shm_ctx();
    TORCH_CHECK_EQ(ctx->group_size, world_size);

    scalar_t* output_ptrs[MAX_SHM_RANK_NUM] = {nullptr};
    for (int i = 0; i < world_size; ++i) {
      output_ptrs[i] = output.data_ptr<scalar_t>() + i * input_elem_num;
    }
    shm_gather_impl(ctx, data.data_ptr<scalar_t>(), data.numel(), output_ptrs,
                    ctx->rank);
    CPU_KERNEL_GUARD_OUT(shm_all_gather_impl)
  });
}

// inputs: this rank's [sizes[rank], ...] tensors; outputs: [sum(sizes), ...]
// tensors receiving every rank's rows in rank order. Any dtype (bytes copy).
void shm_all_gatherv(int64_t handle, const std::vector<torch::Tensor>& inputs,
                     const std::vector<torch::Tensor>& outputs,
                     const std::vector<int64_t>& sizes) {
  CPU_KERNEL_GUARD_IN(shm_all_gatherv)
  shm_all_gatherv_impl(
      SHMManager::get_singleton_instance(handle)->get_shm_ctx(), inputs,
      outputs, sizes);
  CPU_KERNEL_GUARD_OUT(shm_all_gatherv)
}

void shm_allreduce(int64_t handle, torch::Tensor& data) {
  TORCH_CHECK(data.is_contiguous())
  VLLM_DISPATCH_FLOATING_TYPES(data.scalar_type(), "shm_allreduce_sum", [&] {
    CPU_KERNEL_GUARD_IN(shm_allreduce_sum)
    shm_allreduce_sum(SHMManager::get_singleton_instance(handle)->get_shm_ctx(),
                      data.data_ptr<scalar_t>(), data.numel());
    CPU_KERNEL_GUARD_OUT(shm_allreduce_sum)
  });
}

// data: the full tensor on every rank; output: this rank's slice, starting at
// element `out_offset` of data. Without `out_offset` the slices are equal
// sized and ordered by rank (kept out of the schema so that compiled graphs
// don't bake shape-derived offsets in).
void shm_reduce_scatter(int64_t handle, const torch::Tensor& data,
                        torch::Tensor& output,
                        const std::optional<int64_t> out_offset) {
  TORCH_CHECK(data.is_contiguous())
  TORCH_CHECK(output.is_contiguous())
  TORCH_CHECK_EQ(data.scalar_type(), output.scalar_type());
  auto ctx = SHMManager::get_singleton_instance(handle)->get_shm_ctx();
  const int64_t offset = out_offset.has_value()
                             ? out_offset.value()
                             : ctx->rank * output.numel();
  TORCH_CHECK(offset >= 0 && offset + output.numel() <= data.numel());
  VLLM_DISPATCH_FLOATING_TYPES(
      data.scalar_type(), "shm_reduce_scatter_sum", [&] {
        CPU_KERNEL_GUARD_IN(shm_reduce_scatter_sum)
        shm_reduce_scatter_sum(ctx, data.data_ptr<scalar_t>(), data.numel(),
                               output.data_ptr<scalar_t>(), offset,
                               offset + output.numel());
        CPU_KERNEL_GUARD_OUT(shm_reduce_scatter_sum)
      });
}

void shm_send_tensor_list(int64_t handle,
                          const std::vector<torch::Tensor>& tensor_list,
                          int64_t dst) {
  CPU_KERNEL_GUARD_IN(shm_send_tensor_list)
  shm_send_tensor_list_impl(
      SHMManager::get_singleton_instance(handle)->get_shm_ctx(), dst,
      tensor_list);
  CPU_KERNEL_GUARD_OUT(shm_send_tensor_list)
}

std::vector<torch::Tensor> shm_recv_tensor_list(int64_t handle, int64_t src) {
  CPU_KERNEL_GUARD_IN(shm_recv_tensor_list)
  auto tensor_list = shm_recv_tensor_list_impl(
      SHMManager::get_singleton_instance(handle)->get_shm_ctx(), src);
  CPU_KERNEL_GUARD_OUT(shm_recv_tensor_list)
  return tensor_list;
}

int64_t init_shm_manager(const std::string& name, const int64_t group_size,
                         const int64_t rank, const int64_t thread_num) {
  return SHMManager::create_singleton_instance(name, group_size, rank,
                                               thread_num);
}

std::string join_shm_manager(int64_t handle, const std::string& name) {
  auto shm_manager = SHMManager::get_singleton_instance(handle);
  TORCH_CHECK(shm_manager);
  shm_manager->join(name);
  return shm_manager->get_shm_ctx()->to_string();
}
