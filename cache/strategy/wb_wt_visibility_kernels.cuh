#pragma once

#include "ptx_cache_ops.cuh"

namespace {

template <StoreOp kStoreOp>
__device__ __forceinline__ void wb_wt_visibility_body(uint32_t* data,
                                                      uint32_t new_value,
                                                      uint32_t* flags,
                                                      uint32_t* results,
                                                      uint32_t* smids,
                                                      uint64_t producer_delay_cycles) {
  if (threadIdx.x != 0) return;

  if (blockIdx.x == 0) {
    smids[0] = read_smid();
    while (atomicAdd((int*)&flags[0], 0) == 0) {
    }
    (void)asm_ld_ca_u32(data);
    do_store_u32<kStoreOp>(data, new_value);
    delay_cycles(producer_delay_cycles);
    atomicExch((int*)&flags[1], 1);
    return;
  }

  if (blockIdx.x == 1) {
    smids[1] = read_smid();

    uint32_t before = asm_ld_cg_u32(data);
    results[0] = before;
    atomicExch((int*)&flags[0], 1);

    while (atomicAdd((int*)&flags[1], 0) == 0) {
    }

    uint32_t after = asm_ld_cg_u32(data);
    results[1] = after;
    return;
  }
}

} // namespace

#define DEFINE_WB_WT_VISIBILITY_KERNEL(kernel_name, store_op)                    \
  extern "C" __global__ void kernel_name(                                        \
      uint32_t* data, uint32_t new_value, uint32_t* flags, uint32_t* results,    \
      uint32_t* smids, uint64_t producer_delay_cycles) {                         \
    wb_wt_visibility_body<store_op>(                                             \
        data, new_value, flags, results, smids, producer_delay_cycles);          \
  }

DEFINE_WB_WT_VISIBILITY_KERNEL(k_visibility_default, StoreOp::kDefault)
DEFINE_WB_WT_VISIBILITY_KERNEL(k_visibility_wb, StoreOp::kWb)
DEFINE_WB_WT_VISIBILITY_KERNEL(k_visibility_wt, StoreOp::kWt)
DEFINE_WB_WT_VISIBILITY_KERNEL(k_visibility_cg, StoreOp::kCg)
DEFINE_WB_WT_VISIBILITY_KERNEL(k_visibility_cs, StoreOp::kCs)

#undef DEFINE_WB_WT_VISIBILITY_KERNEL
