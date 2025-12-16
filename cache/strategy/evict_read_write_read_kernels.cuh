#pragma once

#include "ptx_cache_ops.cuh"

namespace {

template <StoreOp kStoreOp>
__device__ __forceinline__ uint32_t evict_read_write_read_body(uint32_t* buf,
                                                               int iters,
                                                               int stride_words) {
  uint32_t sum = 0;
  for (int i = 0; i < iters; ++i) {
    uint32_t* p = buf + (size_t)i * (size_t)stride_words;
    uint32_t a = asm_ld_ca_u32(p);
    do_store_u32<kStoreOp>(p, a + 1u);
    sum += asm_ld_ca_u32(p);
  }
  return sum;
}

} // namespace

#define DEFINE_EVICT_RWR_KERNEL(kernel_name, store_op)                      \
  extern "C" __global__ void kernel_name(uint32_t* buf, int iters,          \
                                        int stride_words, uint32_t* out) { \
    if (blockIdx.x == 0 && threadIdx.x == 0) {                              \
      out[0] = evict_read_write_read_body<store_op>(buf, iters,             \
                                                    stride_words);         \
    }                                                                      \
  }

DEFINE_EVICT_RWR_KERNEL(k_evict_default, StoreOp::kDefault)
DEFINE_EVICT_RWR_KERNEL(k_evict_wb, StoreOp::kWb)
DEFINE_EVICT_RWR_KERNEL(k_evict_wt, StoreOp::kWt)
DEFINE_EVICT_RWR_KERNEL(k_evict_cg, StoreOp::kCg)
DEFINE_EVICT_RWR_KERNEL(k_evict_cs, StoreOp::kCs)

#undef DEFINE_EVICT_RWR_KERNEL

