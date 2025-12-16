#include <cuda_runtime.h>
#include <stdint.h>

namespace {
__device__ __forceinline__ int asm_ld_default_s32(const int* p) {
    int v;
    asm volatile("ld.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ int asm_ld_relaxed_sys_s32(const int* p) {
    int v;
    asm volatile("ld.relaxed.sys.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ int asm_ld_relaxed_gpu_s32(const int* p) {
    int v;
    asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ int asm_ld_acquire_sys_s32(const int* p) {
    int v;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ int asm_ld_acquire_gpu_s32(const int* p) {
    int v;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

// 注意：ptxas 接受 ld.global.weak.*，但不能与 .sys 组合（我们之前探测过）
__device__ __forceinline__ int asm_ld_weak_s32(const int* p) {
    int v;
    asm volatile("ld.global.weak.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void asm_st_default_u32(uint32_t* p, uint32_t v) {
    asm volatile("st.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void asm_st_wt_u32(uint32_t* p, uint32_t v) {
    asm volatile("st.global.wt.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void asm_st_relaxed_sys_u32(uint32_t* p, uint32_t v) {
    asm volatile("st.relaxed.sys.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void asm_st_relaxed_gpu_u32(uint32_t* p, uint32_t v) {
    asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void asm_st_release_sys_u32(uint32_t* p, uint32_t v) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void asm_st_release_gpu_u32(uint32_t* p, uint32_t v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
} // namespace

// 单个 kernel 内做完所有 load/store，对比 SASS 时噪声最小（只会有一次参数装载/一次 desc 相关准备）
extern "C" __global__ void k_ldst_relaxed_weak(const int* in, uint32_t* out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int a = asm_ld_default_s32(in);
    int b = asm_ld_relaxed_sys_s32(in);
    int c = asm_ld_relaxed_gpu_s32(in);
    int d = asm_ld_weak_s32(in);
    int e = asm_ld_acquire_sys_s32(in);
    int f = asm_ld_acquire_gpu_s32(in);

    // 用两种 store 形式写回，避免编译器优化掉 load 结果
    asm_st_default_u32(out + 0, (uint32_t)a);
    asm_st_wt_u32(out + 1, (uint32_t)b);
    asm_st_default_u32(out + 2, (uint32_t)c);
    asm_st_wt_u32(out + 3, (uint32_t)d);
    asm_st_relaxed_sys_u32(out + 4, (uint32_t)e);
    asm_st_relaxed_gpu_u32(out + 5, (uint32_t)f);
    asm_st_release_sys_u32(out + 6, (uint32_t)a);
    asm_st_release_gpu_u32(out + 7, (uint32_t)b);
}

