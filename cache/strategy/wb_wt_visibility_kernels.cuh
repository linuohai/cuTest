#pragma once

#include "ptx_cache_ops.cuh"

namespace {

template <StoreOp kStoreOp>
__device__ __forceinline__ void wb_wt_visibility_body(uint32_t* data,
                                                      uint32_t new_value,
                                                      uint32_t* flags,
                                                      uint32_t* results,
                                                      uint32_t* smids,
                                                      const uint32_t* evict_buf,
                                                      int evict_words,
                                                      int evict_stride_words,
                                                      int polls,
                                                      uint64_t producer_delay_cycles) {
  if (threadIdx.x != 0) return;

  if (blockIdx.x == 0) {
    smids[0] = read_smid();
    (void)asm_ld_ca_u32(data);
    do_store_u32<kStoreOp>(data, new_value);
    delay_cycles(producer_delay_cycles);
    atomicExch((int*)&flags[0], 1);

    while (atomicAdd((int*)&flags[1], 0) == 0) {
    }

    uint32_t sink = 0;
    for (int i = 0; i < evict_words; i += evict_stride_words) {
      sink ^= asm_ld_ca_u32(evict_buf + i);
    }
    results[7] = sink;

    atomicExch((int*)&flags[2], 1);
    return;
  }

  if (blockIdx.x == 1) {
    smids[1] = read_smid();

    while (atomicAdd((int*)&flags[0], 0) == 0) {
    }

    uint32_t before_first = asm_ld_cg_u32(data);
    uint32_t before_last = before_first;
    uint32_t before_seen_new = (before_first == new_value) ? 1u : 0u;
    for (int i = 1; i < polls; ++i) {
      uint32_t v = asm_ld_cg_u32(data);
      before_last = v;
      before_seen_new |= (v == new_value) ? 1u : 0u;
    }

    results[0] = before_first;
    results[1] = before_last;
    results[2] = before_seen_new;

    atomicExch((int*)&flags[1], 1);

    while (atomicAdd((int*)&flags[2], 0) == 0) {
    }

    uint32_t after_first = asm_ld_cg_u32(data);
    uint32_t after_last = after_first;
    uint32_t after_seen_new = (after_first == new_value) ? 1u : 0u;
    for (int i = 1; i < polls; ++i) {
      uint32_t v = asm_ld_cg_u32(data);
      after_last = v;
      after_seen_new |= (v == new_value) ? 1u : 0u;
    }

    results[3] = after_first;
    results[4] = after_last;
    results[5] = after_seen_new;
    return;
  }
}

} // namespace

#define DEFINE_WB_WT_VISIBILITY_KERNEL(kernel_name, store_op)                    \
  extern "C" __global__ void kernel_name(                                        \
      uint32_t* data, uint32_t new_value, uint32_t* flags, uint32_t* results,    \
      uint32_t* smids, const uint32_t* evict_buf, int evict_words,               \
      int evict_stride_words, int polls, uint64_t producer_delay_cycles) {       \
    wb_wt_visibility_body<store_op>(                                             \
        data, new_value, flags, results, smids, evict_buf, evict_words,          \
        evict_stride_words, polls, producer_delay_cycles);                       \
  }

DEFINE_WB_WT_VISIBILITY_KERNEL(k_visibility_default, StoreOp::kDefault)
DEFINE_WB_WT_VISIBILITY_KERNEL(k_visibility_wb, StoreOp::kWb)
DEFINE_WB_WT_VISIBILITY_KERNEL(k_visibility_wt, StoreOp::kWt)
DEFINE_WB_WT_VISIBILITY_KERNEL(k_visibility_cg, StoreOp::kCg)
DEFINE_WB_WT_VISIBILITY_KERNEL(k_visibility_cs, StoreOp::kCs)

#undef DEFINE_WB_WT_VISIBILITY_KERNEL

