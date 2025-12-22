CUDA_HOME ?= /home/sunhn/cuda/
NVCC := $(CUDA_HOME)/bin/nvcc
NVDISASM := $(CUDA_HOME)/bin/nvdisasm

# 目标架构列表: V100(70), A100(80), 4090(89)
# 可以通过 `make SMS="80 89"` 覆盖
SMS ?= 70 80 89

# 源文件定义
SRCS := main.cu
HDRS := \
	strategy_types.h \
	ptx_cache_ops.cuh \
	alloc_cross_thread_kernels.cuh \
	alloc_write_read_kernels.cuh \
	store_only_kernels.cuh \
	evict_read_write_read_kernels.cuh \
	wb_wt_visibility_kernels.cuh

# 编译选项
# -lineinfo 用于生成源代码行号信息，方便 nvdisasm 反汇编查看对应关系
COMMON_FLAGS :=  -std=c++17 -lineinfo

# 输出目录
BIN_DIR := bin

# 生成目标列表辅助函数
GEN_TARGETS = $(foreach sm,$(SMS),$(BIN_DIR)/strategybench.sm$(sm)) \
              $(foreach sm,$(SMS),$(BIN_DIR)/strategybench.sm$(sm).ptx) \
              $(foreach sm,$(SMS),$(BIN_DIR)/strategybench.sm$(sm).cubin) \
              $(foreach sm,$(SMS),$(BIN_DIR)/strategybench.sm$(sm).sass)

all: $(GEN_TARGETS)

# 确保输出目录存在
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# 1. 生成可执行文件
$(BIN_DIR)/strategybench.sm%: $(SRCS) $(HDRS) | $(BIN_DIR)
	$(NVCC) $(COMMON_FLAGS) -arch=sm_$* -o $@ $(SRCS)

# 2. 生成 PTX 中间代码
$(BIN_DIR)/strategybench.sm%.ptx: $(SRCS) $(HDRS) | $(BIN_DIR)
	$(NVCC) $(COMMON_FLAGS) -arch=sm_$* -ptx -o $@ $(SRCS)

# 3. 生成 CUBIN 二进制
$(BIN_DIR)/strategybench.sm%.cubin: $(SRCS) $(HDRS) | $(BIN_DIR)
	$(NVCC) $(COMMON_FLAGS) -arch=sm_$* -cubin -o $@ $(SRCS)

# 4. 生成 SASS 反汇编 (包含源码行号信息)
# 使用 -print-line-info (即 -g) 使得 nvdisasm 输出包含源码行信息
$(BIN_DIR)/strategybench.sm%.sass: $(BIN_DIR)/strategybench.sm%.cubin
	$(NVDISASM) -g $< > $@

clean:
	rm -rf $(BIN_DIR)

.PHONY: all clean
