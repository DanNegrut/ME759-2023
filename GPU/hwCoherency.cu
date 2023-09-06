#include <cuda.h>
#include <stdio.h>
#define SZ 8
__global__ void write(int *ret, int a, int b) {
    ret[threadIdx.x] = a + b + threadIdx.x;
}
__global__ void append(int *ret, int a, int b) {
    ret[threadIdx.x] += a + b + threadIdx.x;
}

int main() {
    int *ret;
    cudaMallocManaged(&ret, SZ * sizeof(int));

    // set direct access hint
    cudaMemAdvise(ret, SZ * sizeof(int), cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);  

    // pages populated in GPU memory
    write<<< 1, SZ >>>(ret, 10, 100);            
    cudaDeviceSynchronize();

    // print operation - directManagedMemAccessFromHost=1: CPU accesses GPU memory directly without migrations
    // If directManagedMemAccessFromHost was 0, then CPU faults and triggers device-to-host migration
    for (int i = 0; i < SZ; i++)
        printf("%d: A+B = %d\n", i, ret[i]);        
                                                    
    // directManagedMemAccessFromHost=1: GPU accesses GPU memory without migrations
    // If directManagedMemAccessFromHost was 0, then CPU faults and triggers device-to-host migration
    append <<<1, SZ>>>(ret, 10, 100);            
    cudaDeviceSynchronize(); 
    printf("\nNew results:\n");
    for (int i = 0; i < SZ; i++)
        printf("%d: A+B = %d\n", i, ret[i]);
    cudaFree(ret);
    return 0;
}
