#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "strategy_types.h"

__device__ __forceinline__ uint32_t asm_ld_ca_u32(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ uint32_t asm_ld_cg_u32(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void asm_st_wb_u32(uint32_t* p, uint32_t v) {
  asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void asm_st_wt_u32(uint32_t* p, uint32_t v) {
  asm volatile("st.global.wt.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void asm_st_cg_u32(uint32_t* p, uint32_t v) {
  asm volatile("st.global.cg.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void asm_st_cs_u32(uint32_t* p, uint32_t v) {
  asm volatile("st.global.cs.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

template <StoreOp kStoreOp>
__device__ __forceinline__ void do_store_u32(uint32_t* p, uint32_t v) {
  if constexpr (kStoreOp == StoreOp::kDefault) {
    *p = v;
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

__device__ __forceinline__ uint32_t read_smid() {
  uint32_t smid;
  asm volatile("mov.u32 %0, %smid;" : "=r"(smid));
  return smid;
}

__device__ __forceinline__ void delay_cycles(uint64_t cycles) {
  if (cycles == 0) return;
  uint64_t start = clock64();
  while ((clock64() - start) < cycles) {
  }
}

