#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                  \
  cudaError_t _e = (call);                                      \
  if (_e != cudaSuccess) {                                      \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
            __FILE__, __LINE__, cudaGetErrorString(_e));        \
    std::exit(1);                                               \
  }                                                             \
} while(0)

inline void print_device_info() {
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  printf("GPU: %s, cc %d.%d, SMs=%d, totalGlobalMem=%.2f GiB\n",
         p.name, p.major, p.minor, p.multiProcessorCount,
         (double)p.totalGlobalMem / (1024.0*1024.0*1024.0));
}
