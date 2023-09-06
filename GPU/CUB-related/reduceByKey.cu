#define CUB_STDERR // print CUDA runtime errors to console
#include <stdio.h>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>
#include "test/test_util.h"

using namespace cub;
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

// CustomMin functor
struct CustomMin
{
    template <typename T>
    __host__ __device__ __forceinline__
        T operator()(const T& a, const T& b) const {
        return (b < a) ? b : a;
    }
};

int main(int argc, char** argv) {
    const int num_items = 10;

    // host-side data
    int h_keys_in[num_items] = { 0, 2, 2, 2, 10, 10, 4, 4, 4, 4 };
    int h_values_in[num_items] = { -2, 4, 5, 2, 1, -1, 0, 2, -1, -1 };

    // input data, on the device
    int* d_keys_in = NULL;
    int* d_values_in = NULL;

    // set up device input arrays
    CubDebugExit(g_allocator.DeviceAllocate((void**)& d_keys_in, sizeof(int) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)& d_values_in, sizeof(int) * num_items));

    // set up data on the device
    CubDebugExit(cudaMemcpy(d_keys_in, h_keys_in, sizeof(int) * num_items, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_values_in, h_values_in, sizeof(int) * num_items, cudaMemcpyHostToDevice));

    // allocate device output arrays
    int* d_unique_out = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)& d_unique_out, sizeof(int) * num_items));
    int* d_aggregates_out = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)& d_aggregates_out, sizeof(int) * num_items));
    int* d_num_runs_out = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)& d_num_runs_out, sizeof(int)));

    CustomMin reduction_op;

    // get temporary device storage requirements
    void* d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes, d_keys_in, d_unique_out, d_values_in, d_aggregates_out, d_num_runs_out, reduction_op, num_items);
    // allocate temporary storage
    CubDebugExit(g_allocator.DeviceAllocate((void**)& d_temp_storage, temp_storage_bytes));
    // run the reduce-by-key operation
    cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes, d_keys_in, d_unique_out, d_values_in, d_aggregates_out, d_num_runs_out, reduction_op, num_items);

    // get data back on the host; print out results
    int h_unique_out[num_items];
    int h_aggregates_out[num_items];
    int h_num_runs_out;
    CubDebugExit(cudaMemcpy(&h_num_runs_out, d_num_runs_out, sizeof(int), cudaMemcpyDeviceToHost));
    CubDebugExit(cudaMemcpy(h_unique_out, d_unique_out, sizeof(int) * h_num_runs_out, cudaMemcpyDeviceToHost));
    CubDebugExit(cudaMemcpy(h_aggregates_out, d_aggregates_out, sizeof(int) * h_num_runs_out, cudaMemcpyDeviceToHost));

    for (int i = 0; i < h_num_runs_out; i++)
        std::cout << "i: " << i << "\tKey: " << h_unique_out[i] << "\tAggregate Value: " << h_aggregates_out[i] << std::endl;

    // cleanup
    if (d_keys_in) CubDebugExit(g_allocator.DeviceFree(d_keys_in));
    if (d_values_in) CubDebugExit(g_allocator.DeviceFree(d_values_in));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
    if (d_unique_out) CubDebugExit(g_allocator.DeviceFree(d_unique_out));
    if (d_aggregates_out) CubDebugExit(g_allocator.DeviceFree(d_aggregates_out));
    if (d_num_runs_out) CubDebugExit(g_allocator.DeviceFree(d_num_runs_out));
    return 0;
}
