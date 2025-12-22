#pragma once

#include "ptx_cache_ops.cuh"

namespace {

template <StoreOp kStoreOp>
__device__ __forceinline__ void do_store_u32_asm(uint32_t* p, uint32_t v) {
  if constexpr (kStoreOp == StoreOp::kDefault) {
    asm_st_default_u32(p, v);
  } else if constexpr (kStoreOp == StoreOp::kWb) {
    asm_st_wb_u32(p, v);
  } else if constexpr (kStoreOp == StoreOp::kWt) {
    asm_st_wt_u32(p, v);
  } else if constexpr (kStoreOp == StoreOp::kCg) {
    asm_st_cg_u32(p, v);
  } else if constexpr (kStoreOp == StoreOp::kCs) {
    asm_st_cs_u32(p, v);
  }
}

template <StoreOp kStoreOp>
__device__ __forceinline__ void stst_body(uint32_t* buf,
                                          int iters,
                                          int stride_words,
                                          uint64_t delay) {
  for (int i = 0; i < iters; ++i) {
    uint32_t* p = buf + (size_t)i * (size_t)stride_words;
    do_store_u32_asm<kStoreOp>(p, (uint32_t)i);
    delay_cycles(delay);
    asm_st_default_u32(p, (uint32_t)(i + 1));
  }
}

} // namespace

#define DEFINE_STST_KERNEL(kernel_name, store_op)                           \
  extern "C" __global__ void kernel_name(uint32_t* buf, int iters,          \
                                        int stride_words,                  \
                                        uint64_t delay_cycles,             \
                                        uint32_t* out) {                   \
    if (blockIdx.x == 0 && threadIdx.x == 0) {                              \
      (void)out;                                                           \
      stst_body<store_op>(buf, iters, stride_words, delay_cycles);         \
    }                                                                      \
  }

DEFINE_STST_KERNEL(k_stst_default, StoreOp::kDefault)
DEFINE_STST_KERNEL(k_stst_wb, StoreOp::kWb)
DEFINE_STST_KERNEL(k_stst_wt, StoreOp::kWt)
DEFINE_STST_KERNEL(k_stst_cg, StoreOp::kCg)
DEFINE_STST_KERNEL(k_stst_cs, StoreOp::kCs)

#undef DEFINE_STST_KERNEL
