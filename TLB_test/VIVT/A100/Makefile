CUDA_HOME ?= /home/sunhn/cuda/
NVCC := $(CUDA_HOME)/bin/nvcc

TARGET := bin/tlbbench
SRCS := src/main.cu
INCS := -Isrc

# A100 (Ampere) = sm_80
NVCCFLAGS := -O3 -std=c++17 -arch=sm_80 -lineinfo $(INCS)
# 为了避免编译器做奇怪的跨越优化（尤其是内联 PTX 周围），建议保留默认即可

all: $(TARGET)

$(TARGET): $(SRCS) src/tlb_kernels.cuh src/cuda_utils.cuh
	mkdir -p bin
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRCS)

clean:
	rm -rf bin
