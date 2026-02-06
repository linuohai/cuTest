#include <cuda_runtime.h>

#include <cassert>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr int kBlockThreads = 32;
constexpr int kPrefetchDistanceIters = 8;
constexpr int kCpAsyncStages = 4;

inline void check_cuda(cudaError_t err, const char* what) {
  if (err == cudaSuccess) return;
  std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(err));
  std::exit(1);
}

struct Args {
  int device = 0;
  std::string benchmark = "stride";   // stride | pointer_chase | gather
  // stride: baseline | prefetch_l2 | cp_async
  // pointer_chase: lcg | stride (dependent stride, controlled by --stride-bytes)
  std::string variant = "baseline";

  size_t workset_bytes = 256ull * 1024 * 1024;
  size_t stride_bytes = 128;

  int iters = 100000;
  int repeats = 5;
  int warmup = 1;

  size_t flush_bytes = 256ull * 1024 * 1024;

  double hot_frac = 1.0;   // gather: hotset fraction of data working set
  double hot_prob = 0.9;   // gather: probability of hitting hotset
  uint64_t seed = 1234;

  std::string format = "csv";  // csv | json | human
  bool print_header = false;
};

inline uint64_t parse_u64(const char* s) {
  char* end = nullptr;
  unsigned long long v = std::strtoull(s, &end, 10);
  if (!end || *end != '\0') {
    std::fprintf(stderr, "Invalid u64: %s\n", s);
    std::exit(2);
  }
  return static_cast<uint64_t>(v);
}

inline int parse_i32(const char* s) {
  char* end = nullptr;
  long v = std::strtol(s, &end, 10);
  if (!end || *end != '\0') {
    std::fprintf(stderr, "Invalid i32: %s\n", s);
    std::exit(2);
  }
  return static_cast<int>(v);
}

inline double parse_f64(const char* s) {
  char* end = nullptr;
  double v = std::strtod(s, &end);
  if (!end || *end != '\0') {
    std::fprintf(stderr, "Invalid f64: %s\n", s);
    std::exit(2);
  }
  return v;
}

void print_usage(const char* argv0) {
  std::fprintf(stderr,
               "Usage: %s [--device N] [--benchmark stride|pointer_chase|gather]\n"
               "          [--variant <variant>]\n"
               "          [--workset-bytes BYTES] [--stride-bytes BYTES]\n"
               "          [--iters N] [--repeats N] [--warmup N] [--flush-bytes BYTES]\n"
               "          [--hot-frac F] [--hot-prob P] [--seed S]\n"
               "          [--format csv|json|human] [--print-header]\n",
               argv0);
}

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; i++) {
    std::string_view arg(argv[i]);
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "Missing value for %s\n", argv[i]);
        std::exit(2);
      }
      return argv[++i];
    };

    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else if (arg == "--device") {
      a.device = parse_i32(next());
    } else if (arg == "--benchmark") {
      a.benchmark = next();
    } else if (arg == "--variant") {
      a.variant = next();
    } else if (arg == "--workset-bytes") {
      a.workset_bytes = static_cast<size_t>(parse_u64(next()));
    } else if (arg == "--stride-bytes") {
      a.stride_bytes = static_cast<size_t>(parse_u64(next()));
    } else if (arg == "--iters") {
      a.iters = parse_i32(next());
    } else if (arg == "--repeats") {
      a.repeats = parse_i32(next());
    } else if (arg == "--warmup") {
      a.warmup = parse_i32(next());
    } else if (arg == "--flush-bytes") {
      a.flush_bytes = static_cast<size_t>(parse_u64(next()));
    } else if (arg == "--hot-frac") {
      a.hot_frac = parse_f64(next());
    } else if (arg == "--hot-prob") {
      a.hot_prob = parse_f64(next());
    } else if (arg == "--seed") {
      a.seed = parse_u64(next());
    } else if (arg == "--format") {
      a.format = next();
    } else if (arg == "--print-header") {
      a.print_header = true;
    } else {
      std::fprintf(stderr, "Unknown arg: %s\n", argv[i]);
      print_usage(argv[0]);
      std::exit(2);
    }
  }
  return a;
}

__device__ __forceinline__ void prefetch_global_l2(const void* p) {
  asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}

__device__ __forceinline__ void cp_async_ca_shared_global_4b(unsigned int smem_addr,
                                                             const void* gmem_ptr) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;");
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;" :: "n"(N));
}

__global__ void flush_l2_kernel(const uint64_t* __restrict__ buf, size_t n, uint64_t* out) {
  uint64_t sum = 0;
  size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = tid; i < n; i += stride) {
    sum += buf[i];
  }
  if (tid == 0) out[0] = sum;
}

__global__ void init_lcg_next_kernel(uint32_t* __restrict__ next, uint32_t mask) {
  constexpr uint32_t kA = 1664525u;
  constexpr uint32_t kB = 1013904223u;
  size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  uint64_t n = static_cast<uint64_t>(mask) + 1ull;
  for (uint64_t i = tid; i < n; i += stride) {
    uint32_t x = static_cast<uint32_t>(i);
    next[static_cast<size_t>(i)] = (kA * x + kB) & mask;
  }
}

__global__ void init_stride_next_kernel(uint32_t* __restrict__ next, uint64_t n, uint32_t stride_elems) {
  size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (uint64_t i = tid; i < n; i += stride) {
    uint64_t j = i + static_cast<uint64_t>(stride_elems);
    while (j >= n) j -= n;
    next[static_cast<size_t>(i)] = static_cast<uint32_t>(j);
  }
}

__global__ void stride_baseline_kernel(const int* __restrict__ data,
                                       size_t n,
                                       int stride_elems,
                                       int iters,
                                       int* out) {
  int lane = threadIdx.x & 31;
  if (lane != 0) return;
  size_t idx = 0;
  int acc = 0;
#pragma unroll 1
  for (int t = 0; t < iters; t++) {
    idx += static_cast<size_t>(stride_elems);
    if (idx >= n) idx -= n;
    acc ^= data[idx];
  }
  out[0] = acc;
}

__global__ void stride_prefetch_l2_kernel(const int* __restrict__ data,
                                         size_t n,
                                         int stride_elems,
                                         int iters,
                                         int* out) {
  int lane = threadIdx.x & 31;
  if (lane != 0) return;
  size_t idx = 0;
  int acc = 0;
#pragma unroll 1
  for (int t = 0; t < iters; t++) {
    size_t pf_idx = idx + static_cast<size_t>(stride_elems) * kPrefetchDistanceIters;
    while (pf_idx >= n) pf_idx -= n;
    prefetch_global_l2(data + pf_idx);

    idx += static_cast<size_t>(stride_elems);
    if (idx >= n) idx -= n;
    acc ^= data[idx];
  }
  out[0] = acc;
}

__global__ void stride_cp_async_kernel(const int* __restrict__ data,
                                      size_t n,
                                      int stride_elems,
                                      int iters,
                                      int* out) {
  int lane = threadIdx.x & 31;
  if (lane != 0) return;
  __shared__ int smem[kCpAsyncStages];
  unsigned int smem_base = static_cast<unsigned int>(__cvta_generic_to_shared(smem));

  auto wrap_add = [&](size_t base, int mul) -> size_t {
    size_t idx = base + static_cast<size_t>(stride_elems) * static_cast<size_t>(mul);
    while (idx >= n) idx -= n;
    return idx;
  };

  int acc = 0;
  size_t idx = wrap_add(0, 1);

  // Prologue: preload first kCpAsyncStages iterations.
  // Iteration 0 reads idx = 1*stride, so stage 0 holds that element.
  for (int s = 0; s < kCpAsyncStages; s++) {
    size_t g_idx = wrap_add(0, s + 1);
    cp_async_ca_shared_global_4b(smem_base + static_cast<unsigned int>(s * sizeof(int)),
                                 data + g_idx);
  }
  cp_async_commit_group();
  cp_async_wait_group<0>();
  __syncwarp();

#pragma unroll 1
  for (int t = 0; t < iters; t++) {
    // Stage for iteration t.
    int stage = t % kCpAsyncStages;

    // Wait until the data for this stage is ready.
    cp_async_wait_group<kCpAsyncStages - 1>();
    __syncwarp();

    int v = smem[stage];
    acc ^= v;

    // Enqueue copy for future iteration t + kCpAsyncStages.
    size_t future = wrap_add(idx, kCpAsyncStages);
    cp_async_ca_shared_global_4b(smem_base + static_cast<unsigned int>(stage * sizeof(int)),
                                 data + future);
    cp_async_commit_group();

    idx += static_cast<size_t>(stride_elems);
    if (idx >= n) idx -= n;
  }
  // Drain.
  cp_async_wait_group<0>();
  __syncwarp();
  out[0] = acc;
}

__global__ void pointer_chase_kernel(const uint32_t* __restrict__ next, int iters, int* out) {
  int lane = threadIdx.x & 31;
  if (lane != 0) return;
  uint32_t p = 0;
  uint32_t acc = 0;
#pragma unroll 1
  for (int t = 0; t < iters; t++) {
    p = next[p];
    acc ^= p;
  }
  out[0] = static_cast<int>(acc);
}

__global__ void gather_ima_kernel(const int* __restrict__ data,
                                 int data_n,
                                 const int* __restrict__ indices,
                                 int idx_n,
                                 int iters,
                                 int* out) {
  int lane = threadIdx.x & 31;
  if (lane != 0) return;
  int acc = 0;
  int p = 0;
#pragma unroll 1
  for (int t = 0; t < iters; t++) {
    int j = indices[p];
    acc ^= data[j];
    p++;
    if (p >= idx_n) p = 0;
  }
  out[0] = acc;
}

bool is_pow2(size_t x) {
  return x && ((x & (x - 1)) == 0);
}

}  // namespace

int main(int argc, char** argv) {
  Args a = parse_args(argc, argv);

  if (a.print_header) {
    if (a.format == "csv") {
      std::printf(
          "benchmark,variant,device,workset_bytes,stride_bytes,iters,repeats,hot_frac,hot_prob,"
          "time_ns_per_iter,bandwidth_gbs,checksum\n");
    }
    return 0;
  }

  check_cuda(cudaSetDevice(a.device), "cudaSetDevice");
  cudaDeviceProp prop{};
  check_cuda(cudaGetDeviceProperties(&prop, a.device), "cudaGetDeviceProperties");

  if ((a.stride_bytes % sizeof(int)) != 0) {
    std::fprintf(stderr, "stride-bytes must be multiple of %zu (int).\n", sizeof(int));
    return 2;
  }

  size_t data_n = a.workset_bytes / sizeof(int);
  if (data_n < 2) {
    std::fprintf(stderr, "workset too small.\n");
    return 2;
  }
  // Stride loops using power-of-two strides can accidentally create short cycles when
  // workset is a power-of-two. Use an odd modulus to avoid early wraparound/reuse.
  if ((a.benchmark == "stride") && (data_n % 2 == 0)) {
    data_n -= 1;
  }
  if ((a.benchmark == "pointer_chase") && (a.variant == "stride") && (data_n % 2 == 0)) {
    data_n -= 1;
  }
  int stride_elems = static_cast<int>(a.stride_bytes / sizeof(int));
  if (stride_elems <= 0) stride_elems = 1;

  if (a.benchmark == "pointer_chase") {
    if (a.variant == "baseline" || a.variant == "lcg") {
      if (!is_pow2(data_n)) {
        std::fprintf(stderr, "pointer_chase(lcg) requires power-of-two workset elements.\n");
        return 2;
      }
    } else if (a.variant == "stride") {
      // ok (n is adjusted to odd if needed)
    } else {
      std::fprintf(stderr, "Unknown variant for pointer_chase: %s\n", a.variant.c_str());
      return 2;
    }
    if (data_n > (1ull << 32)) {
      std::fprintf(stderr,
                   "pointer_chase requires <= 2^32 elements (got %" PRIu64 ").\n",
                   static_cast<uint64_t>(data_n));
      return 2;
    }
  }
  if (a.benchmark == "gather") {
    if (data_n > static_cast<size_t>(std::numeric_limits<int>::max())) {
      std::fprintf(stderr,
                   "gather requires workset elements <= INT_MAX (got %" PRIu64 ").\n",
                   static_cast<uint64_t>(data_n));
      return 2;
    }
  }

  // Allocate device buffers.
  int* d_data = nullptr;
  int* d_indices = nullptr;
  uint32_t* d_next = nullptr;
  int* d_out = nullptr;
  uint64_t* d_flush = nullptr;
  uint64_t* d_flush_out = nullptr;

  check_cuda(cudaMalloc(&d_out, sizeof(int)), "cudaMalloc(d_out)");

  // Data buffer used by stride/gather.
  if (a.benchmark != "pointer_chase") {
    check_cuda(cudaMalloc(&d_data, data_n * sizeof(int)), "cudaMalloc(d_data)");
  }

  // Index buffer (pointer chase or gather indices).
  int idx_n = 0;
  if (a.benchmark == "pointer_chase") {
    check_cuda(cudaMalloc(&d_next, data_n * sizeof(uint32_t)), "cudaMalloc(d_next)");
  } else if (a.benchmark == "gather") {
    idx_n = a.iters;
    check_cuda(cudaMalloc(&d_indices, static_cast<size_t>(idx_n) * sizeof(int)),
               "cudaMalloc(d_indices)");
  }

  if (a.flush_bytes > 0) {
    size_t flush_n_u64 = a.flush_bytes / sizeof(uint64_t);
    if (flush_n_u64 > 0) {
      check_cuda(cudaMalloc(&d_flush, flush_n_u64 * sizeof(uint64_t)), "cudaMalloc(d_flush)");
      check_cuda(cudaMalloc(&d_flush_out, sizeof(uint64_t)), "cudaMalloc(d_flush_out)");
      check_cuda(cudaMemset(d_flush, 1, flush_n_u64 * sizeof(uint64_t)), "cudaMemset(d_flush)");
    }
  }

  if (a.benchmark == "pointer_chase") {
    dim3 block(256);
    dim3 grid(240);
    if (a.variant == "baseline" || a.variant == "lcg") {
      uint32_t mask = static_cast<uint32_t>(static_cast<uint64_t>(data_n - 1));
      init_lcg_next_kernel<<<grid, block>>>(d_next, mask);
      check_cuda(cudaGetLastError(), "init_lcg_next_kernel launch");
      check_cuda(cudaDeviceSynchronize(), "init_lcg_next_kernel sync");
    } else if (a.variant == "stride") {
      init_stride_next_kernel<<<grid, block>>>(d_next,
                                               static_cast<uint64_t>(data_n),
                                               static_cast<uint32_t>(stride_elems));
      check_cuda(cudaGetLastError(), "init_stride_next_kernel launch");
      check_cuda(cudaDeviceSynchronize(), "init_stride_next_kernel sync");
    }
  } else if (a.benchmark == "gather") {
    std::mt19937_64 rng(a.seed);
    if (!(a.hot_frac > 0.0 && a.hot_frac <= 1.0)) {
      std::fprintf(stderr, "hot-frac must be in (0, 1].\n");
      return 2;
    }
    if (!(a.hot_prob >= 0.0 && a.hot_prob <= 1.0)) {
      std::fprintf(stderr, "hot-prob must be in [0, 1].\n");
      return 2;
    }
    int hot_n = static_cast<int>(std::max<size_t>(1, static_cast<size_t>(data_n * a.hot_frac)));
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    std::uniform_int_distribution<int> pick_hot(0, hot_n - 1);
    std::uniform_int_distribution<int> pick_all(0, static_cast<int>(data_n - 1));
    std::vector<int> h_indices(static_cast<size_t>(idx_n));
    for (int i = 0; i < idx_n; i++) {
      h_indices[i] = (prob(rng) < a.hot_prob) ? pick_hot(rng) : pick_all(rng);
    }
    check_cuda(cudaMemcpy(d_indices, h_indices.data(), static_cast<size_t>(idx_n) * sizeof(int),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(indices)");
  }

  // Warm-up + timed.
  auto do_flush = [&]() {
    if (!d_flush) return;
    size_t flush_n_u64 = a.flush_bytes / sizeof(uint64_t);
    if (flush_n_u64 == 0) return;
    dim3 block(256);
    dim3 grid(240);
    flush_l2_kernel<<<grid, block>>>(d_flush, flush_n_u64, d_flush_out);
    check_cuda(cudaGetLastError(), "flush launch");
    check_cuda(cudaDeviceSynchronize(), "flush sync");
  };

  auto launch_kernel = [&]() {
    dim3 grid(1);
    dim3 block(kBlockThreads);
    if (a.benchmark == "stride") {
      if (a.variant == "baseline") {
        stride_baseline_kernel<<<grid, block>>>(d_data, data_n, stride_elems, a.iters, d_out);
      } else if (a.variant == "prefetch_l2") {
        stride_prefetch_l2_kernel<<<grid, block>>>(d_data, data_n, stride_elems, a.iters, d_out);
      } else if (a.variant == "cp_async") {
        stride_cp_async_kernel<<<grid, block>>>(d_data, data_n, stride_elems, a.iters, d_out);
      } else {
        std::fprintf(stderr, "Unknown variant for stride: %s\n", a.variant.c_str());
        std::exit(2);
      }
    } else if (a.benchmark == "pointer_chase") {
      pointer_chase_kernel<<<grid, block>>>(d_next, a.iters, d_out);
    } else if (a.benchmark == "gather") {
      gather_ima_kernel<<<grid, block>>>(d_data, static_cast<int>(data_n), d_indices, idx_n, a.iters,
                                         d_out);
    } else {
      std::fprintf(stderr, "Unknown benchmark: %s\n", a.benchmark.c_str());
      std::exit(2);
    }
    check_cuda(cudaGetLastError(), "kernel launch");
  };

  do_flush();
  for (int w = 0; w < a.warmup; w++) {
    launch_kernel();
    check_cuda(cudaDeviceSynchronize(), "warmup sync");
  }

  cudaEvent_t ev_start{}, ev_stop{};
  check_cuda(cudaEventCreate(&ev_start), "event create start");
  check_cuda(cudaEventCreate(&ev_stop), "event create stop");

  float total_ms = 0.0f;
  for (int r = 0; r < a.repeats; r++) {
    do_flush();
    check_cuda(cudaDeviceSynchronize(), "pre-run sync");
    check_cuda(cudaEventRecord(ev_start), "event record start");
    launch_kernel();
    check_cuda(cudaEventRecord(ev_stop), "event record stop");
    check_cuda(cudaEventSynchronize(ev_stop), "event sync stop");
    float ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&ms, ev_start, ev_stop), "event elapsed");
    total_ms += ms;
  }

  check_cuda(cudaEventDestroy(ev_start), "event destroy start");
  check_cuda(cudaEventDestroy(ev_stop), "event destroy stop");

  int h_out = 0;
  check_cuda(cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy(out)");

  double avg_ms = static_cast<double>(total_ms) / static_cast<double>(a.repeats);
  double time_ns_per_iter = (avg_ms * 1e6) / static_cast<double>(a.iters);

  double bytes_per_iter = 0.0;
  if (a.benchmark == "stride") {
    bytes_per_iter = sizeof(int);
  } else if (a.benchmark == "pointer_chase") {
    bytes_per_iter = sizeof(uint32_t);
  } else if (a.benchmark == "gather") {
    bytes_per_iter = 2.0 * sizeof(int);
  }
  double bandwidth_gbs = (bytes_per_iter / 1e9) / (time_ns_per_iter * 1e-9);

  if (a.format == "csv") {
    std::printf("%s,%s,%d,%" PRIu64 ",%" PRIu64 ",%d,%d,%.6f,%.6f,%.3f,%.6f,%d\n",
                a.benchmark.c_str(),
                a.variant.c_str(),
                a.device,
                static_cast<uint64_t>(a.workset_bytes),
                static_cast<uint64_t>(a.stride_bytes),
                a.iters,
                a.repeats,
                a.hot_frac,
                a.hot_prob,
                time_ns_per_iter,
                bandwidth_gbs,
                h_out);
  } else if (a.format == "json") {
    std::printf(
        "{\"benchmark\":\"%s\",\"variant\":\"%s\",\"device\":%d,"
        "\"workset_bytes\":%" PRIu64 ",\"stride_bytes\":%" PRIu64 ","
        "\"iters\":%d,\"repeats\":%d,\"hot_frac\":%.6f,\"hot_prob\":%.6f,"
        "\"time_ns_per_iter\":%.3f,\"bandwidth_gbs\":%.6f,\"checksum\":%d,"
        "\"gpu_name\":\"%s\",\"cc_major\":%d,\"cc_minor\":%d}\n",
        a.benchmark.c_str(),
        a.variant.c_str(),
        a.device,
        static_cast<uint64_t>(a.workset_bytes),
        static_cast<uint64_t>(a.stride_bytes),
        a.iters,
        a.repeats,
        a.hot_frac,
        a.hot_prob,
        time_ns_per_iter,
        bandwidth_gbs,
        h_out,
        prop.name,
        prop.major,
        prop.minor);
  } else {
    std::printf("GPU: %s (cc %d.%d)\n", prop.name, prop.major, prop.minor);
    std::printf("bench=%s variant=%s workset=%" PRIu64 "B stride=%" PRIu64 "B\n",
                a.benchmark.c_str(),
                a.variant.c_str(),
                static_cast<uint64_t>(a.workset_bytes),
                static_cast<uint64_t>(a.stride_bytes));
    std::printf("time: %.3f ns/iter, bw: %.6f GB/s, checksum=%d\n",
                time_ns_per_iter,
                bandwidth_gbs,
                h_out);
  }

  // Free.
  if (d_data) (void)cudaFree(d_data);
  if (d_indices) (void)cudaFree(d_indices);
  if (d_next) (void)cudaFree(d_next);
  if (d_out) (void)cudaFree(d_out);
  if (d_flush) (void)cudaFree(d_flush);
  if (d_flush_out) (void)cudaFree(d_flush_out);
  return 0;
}
