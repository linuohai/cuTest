#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ uint32_t ld_cg_u32(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

__device__ __forceinline__ void st_cs_u32(uint32_t* p, uint32_t v) {
  asm volatile("st.global.cs.u32 [%0], %1;" :: "l"(p), "r"(v));
}

__global__ void streaming_copy_cg_cs(const uint32_t* __restrict__ in,
                                     uint32_t* __restrict__ out,
                                     int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    uint32_t x = ld_cg_u32(in + i);   // 期望看到：ld.global.cg.u32
    st_cs_u32(out + i, x);            // 期望看到：st.global.cs.u32
  }
}
