// Vector addition: C = A + B.

#include <stdio.h>
#include <cuda.h>

// CUDA Kernel Device code
// Computes the vector addition of A and B into C. The 3 vectors have the same
// number of elements numElements.
__global__ void 
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
  // INSERT KERNEL CODE HERE
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    C[idx] = A[idx] + B[idx];
  }
  // END KERNEL CODE
}

// Host main routine
int main(int argc, char** argv) {
    // Set the vector length and compute its size
    int numElements = 50000;
    size_t size = numElements * sizeof(float);

    // Allocate the host vectors A, B, and C
    float* h_A = (float *)malloc(size);
    float* h_B = (float *)malloc(size);
    float* h_C = (float *)malloc(size);

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate the device vectors A, B, and C
    float* d_A = NULL;
    float* d_B = NULL;
    float* d_C = NULL;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy the host input vectors A and B to the device input vectors
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    // Copy the device result vector to the host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify that the result vector is correct
    int i;
    bool correct = true;
    for (i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
          correct = false;
          break;
        }
    }
    if (correct)
      printf("Result is correct.\n");
    else
      printf("Result verification failed at element %d!\n", i);

    // Free device global memory
    cudaFree(d_A);  cudaFree(d_B);  cudaFree(d_C);

    // Free host memory
    free(h_A);  free(h_B);  free(h_C);
}

