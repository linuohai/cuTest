CUDA_ARCH ?= sm_80
NVCC ?= nvcc

CXXFLAGS ?= -O3 -std=c++17 -lineinfo

BIN_DIR := bin
TARGET := $(BIN_DIR)/prefetch_bench

.PHONY: all clean

all: $(TARGET)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(TARGET): prefetch_bench.cu | $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) -arch=$(CUDA_ARCH) $< -o $@

clean:
	@echo "No rm allowed by policy; clean is a no-op."
