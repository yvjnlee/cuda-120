#include <cuda_runtime.h>
#include "kernels.h"

int main() {
	dummyKernel<<<2, 4>>>();
	// cudaDeviceSynchronize blocks CPU from accessing mem until GPU is finished its ops
	cudaDeviceSynchronize();
	return 0;
}

