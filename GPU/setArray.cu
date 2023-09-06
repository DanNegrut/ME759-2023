#include <cuda.h>
#include <stdio.h>

__global__ void simpleKernel(int *data) {
  // this adds a value to a variable stored in global memory
  data[threadIdx.x] += 2 * (blockIdx.x + threadIdx.x);
}

int main() {
  const int numElems = 4;
  int hostArray[numElems], *devArray;
  // allocate memory on the device; zero out all entries in this device array
  cudaMalloc((void **)&devArray, sizeof(int) * numElems);
  cudaMemset(devArray, 0, numElems * sizeof(int));

  // invoke GPU kernel, with one block that has four threads
  simpleKernel<<<1, numElems>>>(devArray);

  // bring the result back from the GPU into the hostArray
  cudaMemcpy(&hostArray, devArray, sizeof(int) * numElems,
             cudaMemcpyDeviceToHost);
  // print out the result to confirm that things are looking good
  printf("Values stored in hostArray: \n");
  for (int i = 0; i < numElems; i++)
    printf("%d\n", hostArray[i]);
  // release the memory allocated on the GPU
  cudaFree(devArray);
  return 0;
}
