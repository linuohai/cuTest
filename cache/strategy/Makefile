CUDA_HOME ?= /home/sunhn/cuda/
NVCC := $(CUDA_HOME)/bin/nvcc

TARGET := bin/strategybench
SRCS := main.cu
HDRS := \
	strategy_types.h \
	ptx_cache_ops.cuh \
	alloc_write_read_kernels.cuh \
	evict_read_write_read_kernels.cuh \
	wb_wt_visibility_kernels.cuh

# 4090 = sm_89；可用 `make SM=80` 之类切换
SM ?= 89

NVCCFLAGS := -O3 -std=c++17 -arch=sm_$(SM) -lineinfo

all: $(TARGET)

$(TARGET): $(SRCS) $(HDRS)
	mkdir -p bin
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRCS)

clean:
	rm -rf bin
