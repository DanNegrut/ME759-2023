#include <cuda.h>
#include <stdio.h>

__global__ void simpleKernel() { printf("Hello World!\n"); }

int main() {
  const int numThreads = 4;

  // invoke GPU kernel, with one block that has four threads
  simpleKernel<<<1, numThreads>>>();
  cudaDeviceSynchronize();
  return 0;
}
