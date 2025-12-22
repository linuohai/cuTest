#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "strategy_types.h"
#include "alloc_cross_thread_kernels.cuh"
#include "alloc_write_read_kernels.cuh"
#include "store_only_kernels.cuh"
#include "evict_read_write_read_kernels.cuh"
#include "wb_wt_visibility_kernels.cuh"

#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t err__ = (call);                                             \
    if (err__ != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(err__));                              \
      std::exit(1);                                                         \
    }                                                                       \
  } while (0)

namespace {

enum class TestKind {
  kAlloc,
  kAlloc2,
  kStst,
  kEvict,
  kVisibility,
};

static void print_usage(const char* argv0) {
  std::fprintf(stderr,
               "Usage:\n"
               "  %s --test <alloc|alloc2|stst|evict|vis> --store <default|wb|wt|cg|cs> [options]\n"
               "\n"
               "Common options:\n"
               "  --iters N            (alloc/alloc2/stst/evict) unique lines touched (default: 1)\n"
               "  --stride_bytes B     (alloc/alloc2/stst/evict) bytes between lines (default: 128)\n"
               "  --info               print device info\n"
               "\n"
               "Visibility (vis) / stst options:\n"
               "  --delay_cycles N     producer delay after store (default: 10000)\n",
               argv0);
}

static bool parse_u64(const char* s, uint64_t* out) {
  if (!s || !*s) return false;
  char* end = nullptr;
  unsigned long long v = std::strtoull(s, &end, 0);
  if (!end || *end != '\0') return false;
  *out = (uint64_t)v;
  return true;
}

static bool parse_i32(const char* s, int* out) {
  if (!s || !*s) return false;
  char* end = nullptr;
  long v = std::strtol(s, &end, 0);
  if (!end || *end != '\0') return false;
  if (v < INT32_MIN || v > INT32_MAX) return false;
  *out = (int)v;
  return true;
}

static bool parse_store_op(const std::string& s, StoreOp* out) {
  if (s == "default") {
    *out = StoreOp::kDefault;
    return true;
  }
  if (s == "wb") {
    *out = StoreOp::kWb;
    return true;
  }
  if (s == "wt") {
    *out = StoreOp::kWt;
    return true;
  }
  if (s == "cg") {
    *out = StoreOp::kCg;
    return true;
  }
  if (s == "cs") {
    *out = StoreOp::kCs;
    return true;
  }
  return false;
}

static bool parse_test_kind(const std::string& s, TestKind* out) {
  if (s == "alloc") {
    *out = TestKind::kAlloc;
    return true;
  }
  if (s == "alloc2") {
    *out = TestKind::kAlloc2;
    return true;
  }
  if (s == "stst") {
    *out = TestKind::kStst;
    return true;
  }
  if (s == "evict") {
    *out = TestKind::kEvict;
    return true;
  }
  if (s == "vis" || s == "visibility") {
    *out = TestKind::kVisibility;
    return true;
  }
  return false;
}

static void print_device_info() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  std::printf("GPU: %s (cc %d.%d)\n", prop.name, prop.major, prop.minor);
  std::printf("SMs: %d\n", prop.multiProcessorCount);
  std::printf("L2:  %zu bytes\n", (size_t)prop.l2CacheSize);
  std::printf("Shared per SM: %zu bytes\n", (size_t)prop.sharedMemPerMultiprocessor);
  std::printf("Shared per block: %zu bytes\n", (size_t)prop.sharedMemPerBlock);
#if CUDART_VERSION >= 9000
  std::printf("Shared per block opt-in: %zu bytes\n",
              (size_t)prop.sharedMemPerBlockOptin);
#endif
}

struct Options {
  TestKind test = TestKind::kAlloc;
  StoreOp store = StoreOp::kDefault;
  int iters = 1;          // 1 * 128B = 128B (smoke test)
  int stride_bytes = 128;
  uint64_t delay_cycles = 10000;
  bool info = false;
};

} // namespace

int main(int argc, char** argv) {
  Options opt;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      print_usage(argv[0]);
      return 0;
    }
    if (a == "--info") {
      opt.info = true;
      continue;
    }
    if (a == "--test" && i + 1 < argc) {
      if (!parse_test_kind(argv[++i], &opt.test)) {
        std::fprintf(stderr, "bad --test: %s\n", argv[i]);
        return 2;
      }
      continue;
    }
    if (a == "--store" && i + 1 < argc) {
      if (!parse_store_op(argv[++i], &opt.store)) {
        std::fprintf(stderr, "bad --store: %s\n", argv[i]);
        return 2;
      }
      continue;
    }
    if (a == "--iters" && i + 1 < argc) {
      if (!parse_i32(argv[++i], &opt.iters) || opt.iters <= 0) {
        std::fprintf(stderr, "bad --iters: %s\n", argv[i]);
        return 2;
      }
      continue;
    }
    if (a == "--stride_bytes" && i + 1 < argc) {
      if (!parse_i32(argv[++i], &opt.stride_bytes) || opt.stride_bytes <= 0) {
        std::fprintf(stderr, "bad --stride_bytes: %s\n", argv[i]);
        return 2;
      }
      continue;
    }
    if (a == "--delay_cycles" && i + 1 < argc) {
      if (!parse_u64(argv[++i], &opt.delay_cycles)) {
        std::fprintf(stderr, "bad --delay_cycles: %s\n", argv[i]);
        return 2;
      }
      continue;
    }

    std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
    print_usage(argv[0]);
    return 2;
  }

  if (opt.info) {
    print_device_info();
  }

  if ((opt.stride_bytes % 4) != 0 || opt.stride_bytes < 4) {
    std::fprintf(stderr, "--stride_bytes must be >=4 and multiple of 4\n");
    return 2;
  }
  if (opt.test == TestKind::kAlloc || opt.test == TestKind::kAlloc2 ||
      opt.test == TestKind::kStst || opt.test == TestKind::kEvict) {
    int stride_words = opt.stride_bytes / 4;
    size_t buf_bytes = (size_t)opt.iters * (size_t)opt.stride_bytes + sizeof(uint32_t);

    uint32_t* d_buf = nullptr;
    uint32_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, buf_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_buf, 0, buf_bytes));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(uint32_t)));

    dim3 grid(1);
    dim3 block(1);

    if (opt.test == TestKind::kAlloc) {
      switch (opt.store) {
        case StoreOp::kDefault: k_alloc_default<<<grid, block>>>(d_buf, opt.iters, stride_words, d_out); break;
        case StoreOp::kWb: k_alloc_wb<<<grid, block>>>(d_buf, opt.iters, stride_words, d_out); break;
        case StoreOp::kWt: k_alloc_wt<<<grid, block>>>(d_buf, opt.iters, stride_words, d_out); break;
        case StoreOp::kCg: k_alloc_cg<<<grid, block>>>(d_buf, opt.iters, stride_words, d_out); break;
        case StoreOp::kCs: k_alloc_cs<<<grid, block>>>(d_buf, opt.iters, stride_words, d_out); break;
      }
    } else if (opt.test == TestKind::kAlloc2) {
      block = dim3(2);
      switch (opt.store) {
        case StoreOp::kDefault: k_alloc2_default<<<grid, block>>>(d_buf, opt.iters, stride_words, d_out); break;
        case StoreOp::kWb: k_alloc2_wb<<<grid, block>>>(d_buf, opt.iters, stride_words, d_out); break;
        case StoreOp::kWt: k_alloc2_wt<<<grid, block>>>(d_buf, opt.iters, stride_words, d_out); break;
        case StoreOp::kCg: k_alloc2_cg<<<grid, block>>>(d_buf, opt.iters, stride_words, d_out); break;
        case StoreOp::kCs: k_alloc2_cs<<<grid, block>>>(d_buf, opt.iters, stride_words, d_out); break;
      }
    } else if (opt.test == TestKind::kStst) {
      switch (opt.store) {
        case StoreOp::kDefault: k_stst_default<<<grid, block>>>(d_buf, opt.iters, stride_words, opt.delay_cycles, d_out); break;
        case StoreOp::kWb: k_stst_wb<<<grid, block>>>(d_buf, opt.iters, stride_words, opt.delay_cycles, d_out); break;
        case StoreOp::kWt: k_stst_wt<<<grid, block>>>(d_buf, opt.iters, stride_words, opt.delay_cycles, d_out); break;
        case StoreOp::kCg: k_stst_cg<<<grid, block>>>(d_buf, opt.iters, stride_words, opt.delay_cycles, d_out); break;
        case StoreOp::kCs: k_stst_cs<<<grid, block>>>(d_buf, opt.iters, stride_words, opt.delay_cycles, d_out); break;
      }
    } else {
      switch (opt.store) {
        case StoreOp::kDefault: k_evict_default<<<grid, block>>>(d_buf, opt.iters, stride_words, d_out); break;
        case StoreOp::kWb: k_evict_wb<<<grid, block>>>(d_buf, opt.iters, stride_words, d_out); break;
        case StoreOp::kWt: k_evict_wt<<<grid, block>>>(d_buf, opt.iters, stride_words, d_out); break;
        case StoreOp::kCg: k_evict_cg<<<grid, block>>>(d_buf, opt.iters, stride_words, d_out); break;
        case StoreOp::kCs: k_evict_cs<<<grid, block>>>(d_buf, opt.iters, stride_words, d_out); break;
      }
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (opt.test == TestKind::kAlloc2) {
      std::printf("alloc2: done (mismatch not reported)\n");
    } else {
      uint32_t out = 0;
      CUDA_CHECK(cudaMemcpy(&out, d_out, sizeof(uint32_t), cudaMemcpyDeviceToHost));
      std::printf("ok: out=%u (ignore; use Nsight for hit-rate)\n", out);
    }

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_buf));
    return 0;
  }

  if (opt.test == TestKind::kVisibility) {
    uint32_t* d_data = nullptr;
    uint32_t* d_flags = nullptr;
    uint32_t* d_results = nullptr;
    uint32_t* d_smids = nullptr;

    CUDA_CHECK(cudaMalloc(&d_data, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_flags, 2 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_results, 2 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_smids, 2 * sizeof(uint32_t)));

    uint32_t init = 0x12345678u;
    CUDA_CHECK(cudaMemcpy(d_data, &init, sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_flags, 0, 2 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_results, 0, 2 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_smids, 0, 2 * sizeof(uint32_t)));

    uint32_t new_value = 0xdeadbeefu;

    dim3 grid(2);
    dim3 block(1);

    switch (opt.store) {
      case StoreOp::kDefault:
        k_visibility_default<<<grid, block>>>(d_data, new_value, d_flags, d_results, d_smids,
                                             opt.delay_cycles);
        break;
      case StoreOp::kWb:
        k_visibility_wb<<<grid, block>>>(d_data, new_value, d_flags, d_results, d_smids,
                                        opt.delay_cycles);
        break;
      case StoreOp::kWt:
        k_visibility_wt<<<grid, block>>>(d_data, new_value, d_flags, d_results, d_smids,
                                        opt.delay_cycles);
        break;
      case StoreOp::kCg:
        k_visibility_cg<<<grid, block>>>(d_data, new_value, d_flags, d_results, d_smids,
                                        opt.delay_cycles);
        break;
      case StoreOp::kCs:
        k_visibility_cs<<<grid, block>>>(d_data, new_value, d_flags, d_results, d_smids,
                                        opt.delay_cycles);
        break;
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t smids[2]{};
    uint32_t results[2]{};
    CUDA_CHECK(cudaMemcpy(smids, d_smids, sizeof(smids), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(results, d_results, sizeof(results), cudaMemcpyDeviceToHost));

    std::printf("smid: producer=%u consumer=%u%s\n", smids[0], smids[1],
                (smids[0] == smids[1]) ? " (WARNING: same SM; L1 may be shared)" : "");
    std::printf("data init=0x%08x new=0x%08x\n", init, new_value);
    std::printf("before: value=0x%08x seen_new=%u\n", results[0],
                (results[0] == new_value) ? 1u : 0u);
    std::printf("after:  value=0x%08x seen_new=%u\n", results[1],
                (results[1] == new_value) ? 1u : 0u);

    CUDA_CHECK(cudaFree(d_smids));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFree(d_flags));
    CUDA_CHECK(cudaFree(d_data));
    return 0;
  }

  return 0;
}
