#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// 用内联 PTX 强制选择缓存策略（ca: cache in L1+L2; cg: cache in L2, bypass L1）
__device__ __forceinline__ uint64_t ld_ca_u64(const uint64_t* addr) {
  uint64_t out;
  asm volatile("ld.global.ca.u64 %0, [%1];" : "=l"(out) : "l"(addr) : "memory");
  return out;
}
__device__ __forceinline__ uint64_t ld_cg_u64(const uint64_t* addr) {
  uint64_t out;
  asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(out) : "l"(addr) : "memory");
  return out;
}

__device__ __forceinline__ uint64_t clock64_read() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t));
  return t;
}

// 在 base + i*stride 位置写入“下一跳的 device VA”
// 形成一个环：node[i] -> node[(i+step)%n]
// 要保证单一大环覆盖所有节点，需要 gcd(step, n) == 1
__global__ void init_ring(uint8_t* base, uint64_t bytes, uint64_t stride, uint64_t step) {
  uint64_t n = bytes / stride;
  uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;

  uint64_t next = (tid + step) % n;
  uint8_t* cur_ptr  = base + tid  * stride;
  uint8_t* next_ptr = base + next * stride;

  // 每个节点只写一个 64-bit 指针值（写在节点起始处）
  *reinterpret_cast<uint64_t*>(cur_ptr) = reinterpret_cast<uint64_t>(next_ptr);
}

// pointer-chase：先 warmup_rounds 次（不计时）再 timed_rounds 次（计时）
// 只用 thread0 执行，避免并行隐藏延迟
__global__ void chase(const uint8_t* base,
                      uint64_t bytes, uint64_t stride,
                      int warmup_rounds, int timed_rounds,
                      int use_ca,  // 1: ca, 0: cg
                      uint64_t* out_cycles,
                      uint64_t* out_steps,
                      uint64_t* out_sink) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  uint64_t n = bytes / stride;
  if (n < 2) { // 不足以形成有效环
    out_cycles[0] = 0;
    out_steps[0]  = 0;
    out_sink[0]   = 0;
    return;
  }

  uint64_t p = reinterpret_cast<uint64_t>(base);

  // warmup：建立稳定状态（cache/TLB 状态）
  if (use_ca) {
    for (int r = 0; r < warmup_rounds; r++) {
      #pragma unroll 1
      for (uint64_t i = 0; i < n; i++) {
        p = ld_ca_u64(reinterpret_cast<const uint64_t*>(p));
      }
    }

    uint64_t start = clock64_read();
    for (int r = 0; r < timed_rounds; r++) {
      #pragma unroll 1
      for (uint64_t i = 0; i < n; i++) {
        p = ld_ca_u64(reinterpret_cast<const uint64_t*>(p));
      }
    }
    uint64_t stop = clock64_read();

    out_cycles[0] = stop - start;
    out_steps[0]  = (uint64_t)timed_rounds * n;
    out_sink[0]   = p; // side-effect，防止优化
    return;
  } else {
    for (int r = 0; r < warmup_rounds; r++) {
      #pragma unroll 1
      for (uint64_t i = 0; i < n; i++) {
        p = ld_cg_u64(reinterpret_cast<const uint64_t*>(p));
      }
    }

    uint64_t start = clock64_read();
    for (int r = 0; r < timed_rounds; r++) {
      #pragma unroll 1
      for (uint64_t i = 0; i < n; i++) {
        p = ld_cg_u64(reinterpret_cast<const uint64_t*>(p));
      }
    }
    uint64_t stop = clock64_read();

    out_cycles[0] = stop - start;
    out_steps[0]  = (uint64_t)timed_rounds * n;
    out_sink[0]   = p; // side-effect，防止优化
    return;
  }

}
