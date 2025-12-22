#pragma once

#include "ptx_cache_ops.cuh"

namespace {

template <StoreOp kStoreOp>
__device__ __forceinline__ uint32_t alloc_write_then_read_body(uint32_t* buf,
                                                               int iters,
                                                               int stride_words) {
  uint32_t sum = 0;
  constexpr int kLoadOffsetWords = 0;  // same line, different word -> avoid store-to-load forwarding
  for (int i = 0; i < iters; ++i) {
    uint32_t* p = buf + (size_t)i * (size_t)stride_words;
    do_store_u32<kStoreOp>(p, (uint32_t)i);
    sum += asm_ld_ca_u32(p + kLoadOffsetWords);
  }
  return sum;
}

} // namespace

#define DEFINE_ALLOC_WRITE_READ_KERNEL(kernel_name, store_op)               \
  extern "C" __global__ void kernel_name(uint32_t* buf, int iters,          \
                                        int stride_words, uint32_t* out) { \
    __shared__ volatile uint32_t sink;                                     \
    if (blockIdx.x == 0 && threadIdx.x == 0) {                              \
      (void)out;                                                           \
      uint32_t sum = alloc_write_then_read_body<store_op>(buf, iters,       \
                                                         stride_words);    \
      sink = sum;                                                          \
    }                                                                      \
  }

DEFINE_ALLOC_WRITE_READ_KERNEL(k_alloc_default, StoreOp::kDefault)
DEFINE_ALLOC_WRITE_READ_KERNEL(k_alloc_wb, StoreOp::kWb)
DEFINE_ALLOC_WRITE_READ_KERNEL(k_alloc_wt, StoreOp::kWt)
DEFINE_ALLOC_WRITE_READ_KERNEL(k_alloc_cg, StoreOp::kCg)
DEFINE_ALLOC_WRITE_READ_KERNEL(k_alloc_cs, StoreOp::kCs)

#undef DEFINE_ALLOC_WRITE_READ_KERNEL
