// Author: Ruochun Zhang

#include <cstdio>
#include <cuda.h>

const unsigned int nThreads = 8;

__global__ void hello_from_cuda() {
    if (threadIdx.x < nThreads) {
        printf("Hello, I am thread %u\n", threadIdx.x);
    }
}

int main(int argc, char* argv[]) {
    hello_from_cuda<<<1, nThreads>>>();
    cudaDeviceSynchronize();
    return 0;
}
