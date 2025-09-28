#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloFromGPU() {
    printf("Hello from the GPU!\n");
}

int main() {
    helloFromGPU<<<1, 1>>>();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Device sync failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Hello from the CPU!\n");
    return 0;
}

