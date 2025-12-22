#pragma once

#include "ptx_cache_ops.cuh"

namespace {

struct Alloc2Result {
  uint32_t sum;
  uint32_t mismatch;
};

template <StoreOp kStoreOp>
__device__ __forceinline__ Alloc2Result alloc_store_cross_thread_body(uint32_t* buf,
                                                                      int iters,
                                                                      int stride_words) {
  Alloc2Result r{0u, 0u};
  for (int i = 0; i < iters; ++i) {
    uint32_t* p = buf + (size_t)i * (size_t)stride_words;
    if (threadIdx.x == 0) {
      do_store_u32<kStoreOp>(p, (uint32_t)i);
    }
    __syncthreads();
    if (threadIdx.x == 1) {
      uint32_t v = asm_ld_cg_u32(p);
      r.sum += v;
      r.mismatch += (v != (uint32_t)i);
    }
    __syncthreads();
  }
  return r;
}

} // namespace

#define DEFINE_ALLOC_CROSS_THREAD_KERNEL(kernel_name, store_op)            \
  extern "C" __global__ void kernel_name(uint32_t* buf, int iters,          \
                                        int stride_words,                 \
                                        uint32_t* out) {                  \
    __shared__ volatile uint32_t sink;                                     \
    Alloc2Result r = alloc_store_cross_thread_body<store_op>(buf, iters,   \
                                                            stride_words);\
    if (threadIdx.x == 1) {                                                \
      (void)out;                                                           \
      sink = r.sum ^ r.mismatch;                                           \
    }                                                                      \
  }

DEFINE_ALLOC_CROSS_THREAD_KERNEL(k_alloc2_default, StoreOp::kDefault)
DEFINE_ALLOC_CROSS_THREAD_KERNEL(k_alloc2_wb, StoreOp::kWb)
DEFINE_ALLOC_CROSS_THREAD_KERNEL(k_alloc2_wt, StoreOp::kWt)
DEFINE_ALLOC_CROSS_THREAD_KERNEL(k_alloc2_cg, StoreOp::kCg)
DEFINE_ALLOC_CROSS_THREAD_KERNEL(k_alloc2_cs, StoreOp::kCs)

#undef DEFINE_ALLOC_CROSS_THREAD_KERNEL
