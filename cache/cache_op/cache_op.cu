#include <cuda_runtime.h>
#include <stdint.h>

// 目标：用不同写法生成带 cache op 的 PTX 指令（便于你直接看 *.ptx / *.sass）
//
// 覆盖的 PTX cache op（global memory）：
// - load : ld.global.{ca,cg,cs,lu,cv} + ld.global.nc(__ldg)
// - store: st.global.{wb,cg,cs,wt}
//
// 生成方式：
// 1) 直接写 inline PTX asm（最直接，PTX 里必然出现）
// 2) 用 CUDA 内建 intrinsic（__ldca/__ldcg/.../__stwb/__stcg/.../__ldg）
// 3) 对比：C++ 的 volatile（会变成 ld.volatile/st.volatile，并不是 ld.global.cv）

namespace {

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

__device__ __forceinline__ uint32_t asm_ld_cs_u32(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ uint32_t asm_ld_lu_u32(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.lu.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ uint32_t asm_ld_cv_u32(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ uint32_t asm_ld_nc_u32(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void asm_st_wb_u32(uint32_t* p, uint32_t v) {
  asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void asm_st_cg_u32(uint32_t* p, uint32_t v) {
  asm volatile("st.global.cg.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void asm_st_cs_u32(uint32_t* p, uint32_t v) {
  asm volatile("st.global.cs.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void asm_st_wt_u32(uint32_t* p, uint32_t v) {
  asm volatile("st.global.wt.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

} // namespace

extern "C" __global__ void k_asm_all_load_ops(const uint32_t* __restrict__ in,
                                             uint32_t* __restrict__ out,
                                             int n) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;

  // 期望在 PTX 看到：
  // ld.global.{ca,cg,cs,lu,cv,nc}.u32
  uint32_t a = asm_ld_ca_u32(in + i);
  uint32_t b = asm_ld_cg_u32(in + i);
  uint32_t c = asm_ld_cs_u32(in + i);
  uint32_t d = asm_ld_lu_u32(in + i);
  uint32_t e = asm_ld_cv_u32(in + i);
  uint32_t f = asm_ld_nc_u32(in + i);

  out[i] = (a + 3u * b) ^ (5u * c) ^ (7u * d) ^ (11u * e) ^ (13u * f);
}

extern "C" __global__ void k_intrinsic_all_load_ops(const uint32_t* __restrict__ in,
                                                   uint32_t* __restrict__ out,
                                                   int n) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;

  // 期望在 PTX 看到（这些 intrinsic 本质上就是头文件里的 inline asm）：
  // __ldca -> ld.global.ca
  // __ldcg -> ld.global.cg
  // __ldcs -> ld.global.cs
  // __ldlu -> ld.global.lu
  // __ldcv -> ld.global.cv
  // __ldg  -> ld.global.nc
  uint32_t a = __ldca(in + i);
  uint32_t b = __ldcg(in + i);
  uint32_t c = __ldcs(in + i);
  uint32_t d = __ldlu(in + i);
  uint32_t e = __ldcv(in + i);
  uint32_t f = __ldg(in + i);

  out[i] = (a + b) ^ (c + d) ^ (e + 13u * f);
}

extern "C" __global__ void k_asm_all_store_ops(const uint32_t* __restrict__ in,
                                              uint32_t* __restrict__ out_wb,
                                              uint32_t* __restrict__ out_cg,
                                              uint32_t* __restrict__ out_cs,
                                              uint32_t* __restrict__ out_wt,
                                              int n) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;

  uint32_t v = in[i] + (uint32_t)i;
  // 期望在 PTX 看到：st.global.{wb,cg,cs,wt}.u32
  asm_st_wb_u32(out_wb + i, v);
  asm_st_cg_u32(out_cg + i, v);
  asm_st_cs_u32(out_cs + i, v);
  asm_st_wt_u32(out_wt + i, v);
}

extern "C" __global__ void k_intrinsic_all_store_ops(const uint32_t* __restrict__ in,
                                                    uint32_t* __restrict__ out_wb,
                                                    uint32_t* __restrict__ out_cg,
                                                    uint32_t* __restrict__ out_cs,
                                                    uint32_t* __restrict__ out_wt,
                                                    int n) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;

  uint32_t v = in[i] + 17u * (uint32_t)i;
  // 期望在 PTX 看到（同样来自头文件 inline asm）：
  // __stwb -> st.global.wb
  // __stcg -> st.global.cg
  // __stcs -> st.global.cs
  // __stwt -> st.global.wt
  __stwb(out_wb + i, v);
  __stcg(out_cg + i, v);
  __stcs(out_cs + i, v);
  __stwt(out_wt + i, v);
}

extern "C" __global__ void k_cpp_volatile_load_store(const uint32_t* __restrict__ in,
                                                    uint32_t* __restrict__ out,
                                                    int n) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;

  // 对比：C++ volatile 生成的是 ld.volatile / st.volatile（不是 ld.global.cv）
  volatile const uint32_t* vin = in;
  volatile uint32_t* vout = out;
  vout[i] = vin[i];
}

extern "C" __global__ void k_cpp_default_load_store(const uint32_t* __restrict__ in,
                                                   uint32_t* __restrict__ out,
                                                   int n) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;

  // baseline：一般会生成 ld.global / st.global（不带 cache op）
  out[i] = in[i];
}
