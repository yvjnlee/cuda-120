#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);

    if (err != cudaSuccess) {
        printf("cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("CUDA device count: %d\n", count);

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: %s, compute capability %d.%d\n",
               i, prop.name, prop.major, prop.minor);
    }

    return 0;
}

