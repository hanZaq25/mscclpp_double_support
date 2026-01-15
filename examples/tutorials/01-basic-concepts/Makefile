CUDA_HOME ?= /usr/local/cuda
ROCM_HOME ?= /opt/rocm
MSCCLPP_HOME ?= $(HOME)/mscclpp-install

# Check if nvcc exists, otherwise use hipcc
ifeq ($(shell which $(CUDA_HOME)/bin/nvcc 2>/dev/null),)
    COMPILER := $(ROCM_HOME)/bin/hipcc
    ARCH_FLAG := -D__HIP_PLATFORM_AMD__=1
    RPATH_FLAG := -Wl,-rpath,$(MSCCLPP_HOME)/lib
else
    COMPILER := $(CUDA_HOME)/bin/nvcc
    ARCH_FLAG := -arch=native
    RPATH_FLAG := -Xlinker -rpath -Xlinker $(MSCCLPP_HOME)/lib
endif

TARGET = gpu_ping_pong
SRC = gpu_ping_pong.cu
INCLUDE_FLAGS := -I$(MSCCLPP_HOME)/include
LIBRARY_FLAGS := -L$(MSCCLPP_HOME)/lib

all: $(TARGET)

$(TARGET): $(SRC)
	$(COMPILER) $(ARCH_FLAG) $(INCLUDE_FLAGS) $(LIBRARY_FLAGS) $(RPATH_FLAG) -o $@ $< -lmscclpp

clean:
	rm -f $(TARGET)