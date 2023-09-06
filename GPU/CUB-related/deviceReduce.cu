#define CUB_STDERR // print CUDA runtime errors to console
#include <stdio.h>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>
#include "cub/util_debug.cuh"
using namespace cub;
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

int main() {
    const size_t num_items = 10;
    // Set up host arrays
    int h_in[num_items] = { 2, 3, -1, 0, 3, 6, 7, 2, -2, 0 };
    int  sum = 0;
    for (unsigned int i = 0; i < num_items; i++)
        sum += h_in[i];

    // Set up device arrays
    int* d_in = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)& d_in, sizeof(int) * num_items));
    // Initialize device input
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(int) * num_items, cudaMemcpyHostToDevice));
    // Setup device output array
    int* d_sum = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)& d_sum, sizeof(int) * 1));
    // Request and allocate temporary storage
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, num_items));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Do the actual reduce operation
    CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, num_items));
    int gpu_sum;
    CubDebugExit(cudaMemcpy(&gpu_sum, d_sum, sizeof(int) * 1, cudaMemcpyDeviceToHost));
    // Check for correctness
    printf("\t%s\n", (gpu_sum == sum ? "Test passed." : "Test falied."));
    printf("\tSum is: %d\n", gpu_sum);

    // Cleanup
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
    if (d_sum) CubDebugExit(g_allocator.DeviceFree(d_sum));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
    
    return 0;
}
