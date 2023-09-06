#define CUB_STDERR // print CUDA runtime errors to console
#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>
#include "test/test_util.h"

using namespace cub;

CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

struct pairsOfBodies {
    unsigned int body_I;
    unsigned int body_J;
};

int main() {
    const unsigned int num_items = 8;

    // host side setup
    pairsOfBodies h_vals[num_items];
    unsigned int h_keys[num_items] = { 2 , 0, 7, 3, 5, 4, 1, 6 };
    pairsOfBodies h_vals_out[num_items];
    unsigned int h_keys_out[num_items];
    h_vals[0].body_I = 3; h_vals[0].body_J = 0;
    h_vals[1].body_I = 9; h_vals[1].body_J = 2;
    h_vals[2].body_I = 0; h_vals[2].body_J = 9;
    h_vals[3].body_I = 2; h_vals[3].body_J = 4;
    h_vals[4].body_I = 1; h_vals[4].body_J = 5;
    h_vals[5].body_I = 1; h_vals[5].body_J = 7;
    h_vals[6].body_I = 2; h_vals[6].body_J = 9;
    h_vals[7].body_I = 6; h_vals[7].body_J = 8;

    // device side setup
    unsigned int* d_keysIN = NULL;
    unsigned int* d_keysOUT = NULL;
    pairsOfBodies* d_valsIN = NULL;
    pairsOfBodies* d_valsOUT = NULL;

    CubDebugExit(g_allocator.DeviceAllocate((void**)& d_keysIN, sizeof(unsigned int) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)& d_keysOUT, sizeof(unsigned int) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)& d_valsIN, sizeof(pairsOfBodies) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)& d_valsOUT, sizeof(pairsOfBodies) * num_items));

    // get memory set aside on the device
    size_t  temp_storage_bytes = 0;
    void* d_temp_storage = NULL;
    CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keysIN, d_keysOUT, d_valsIN, d_valsOUT, num_items));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
    // initialize data on the device
    CubDebugExit(cudaMemcpy(d_keysIN, h_keys, sizeof(unsigned int) * num_items, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_valsIN, h_vals, sizeof(pairsOfBodies) * num_items, cudaMemcpyHostToDevice));
    // do the actual sort
    CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keysIN, d_keysOUT, d_valsIN, d_valsOUT, num_items));

    // get data back
    CubDebugExit(cudaMemcpy(h_keys_out, d_keysOUT, sizeof(unsigned int) * num_items, cudaMemcpyDeviceToHost));
    CubDebugExit(cudaMemcpy(h_vals_out, d_valsOUT, sizeof(pairsOfBodies) * num_items, cudaMemcpyDeviceToHost));

    // clean up
    if (d_keysIN) CubDebugExit(g_allocator.DeviceFree(d_keysIN));
    if (d_keysOUT) CubDebugExit(g_allocator.DeviceFree(d_keysOUT));
    if (d_valsIN) CubDebugExit(g_allocator.DeviceFree(d_valsIN));
    if (d_valsOUT) CubDebugExit(g_allocator.DeviceFree(d_valsOUT));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    return 0;
}