#include <stdio.h>
#include "kernels.h"

__global__ void dummyKernel() {
	printf("block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

