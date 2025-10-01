PROJECT = cuda_app
NVCC = nvcc
SRCS = src/main.cu src/kernels.cu
INCLUDES = -I./include
ARCH = -arch=sm_61

all: $(PROJECT)

$(PROJECT): $(SRC)
	$(NVCC) $(ARCH) $(INCLUDES) -o $(PROJECT) $(SRCS)

clean:
	rm -f $(PROJECT) *.o

