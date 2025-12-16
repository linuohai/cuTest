CUDA_HOME ?= /home/sunhn/cuda
NVCC := $(CUDA_HOME)/bin/nvcc
NVDISASM := $(CUDA_HOME)/bin/nvdisasm
CUOBJDUMP := $(CUDA_HOME)/bin/cuobjdump

# 4090 = sm_89；可用 `make SM=80` 之类切换
SM ?= 89

SRC := cache_op.cu
OUT := cache_op.sm$(SM)

NVCCFLAGS := -O3 -std=c++17 -arch=sm_$(SM) -lineinfo

.PHONY: all ptx cubin sass sass_cuobjdump check clean

all: ptx cubin sass

ptx: $(OUT).ptx
cubin: $(OUT).cubin
sass: $(OUT).sass

$(OUT).ptx: $(SRC)
	$(NVCC) $(NVCCFLAGS) -ptx -o $@ $<

$(OUT).cubin: $(SRC)
	$(NVCC) $(NVCCFLAGS) -cubin -o $@ $<

$(OUT).sass: $(OUT).cubin
	$(NVDISASM) -g $< > $@

sass_cuobjdump: $(OUT).cubin
	$(CUOBJDUMP) --dump-sass $< > $(OUT).cuobjdump.sass

check: $(OUT).ptx
	@echo "== PTX: ld/st with cache op =="
	@grep -nE "ld\\.(volatile\\.)?global\\.|st\\.(volatile\\.)?global\\." $(OUT).ptx || true

clean:
	rm -f $(OUT).ptx $(OUT).cubin $(OUT).sass $(OUT).cuobjdump.sass

