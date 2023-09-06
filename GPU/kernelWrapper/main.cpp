#include <iostream>
#include <vector>

// Exposes the CUDA Runtime API
#include <cuda_runtime_api.h>

// Provides a Managed Unified Memory allocator
#include "cudalloc.hpp"
#include "kern.h"

int main(int argc, char **argv) {
	
	const size_t N = 1024;
	
	// Managed argument vectors
	std::vector<int, cudallocator<int>> ad(N, 0);
	std::vector<int, cudallocator<int>> bd(N, 0);
	
	// Managed result vector
	std::vector<int, cudallocator<int>> cd(N, 0);
	
	// Host argument vectors
	std::vector<int> ah(N, 0);
	std::vector<int> bh(N, 0);
	
	// Host result vector
	std::vector<int> ch(N, 0);
	
	
	// Populate the vector with some values which are easy to eyeball
	for (int i = 0; i < N; i++) {
		ad[i] = i;
		ah[i] = i;
		bd[i] = i * 2;
		bh[i] = i * 2;
	}
	
	int* ad_ptr = ad.data();
	int* bd_ptr = bd.data();
	int* cd_ptr = cd.data();
	
	void* params[] = {
		&(ad_ptr),
		&(bd_ptr),
		&(cd_ptr)
	};

	// First, launch the kernel
	cudaError_t status = cudaSuccess;
	if ((status = 
		cudaLaunchKernel(
			KERNEL_HANDLE(add_vector), 
			dim3(1,1,1), 
			dim3(1024, 1, 1), 
			params, 
			0, 
			cudaStreamLegacy)) != cudaSuccess) {
		std::cout << "Kernel launch failed with error: " << status << "\n";
		return 1;
	}
	
	for (size_t i = 0; i < N; i++) {
		ch[i] = ah[i] + bh[i];
	}
	
	// Explicitly synchronize with the kernel stream
    status = cudaStreamSynchronize(cudaStreamLegacy);
	if (status != cudaSuccess) {
		std::cout << "Stream synchronize failed with error: " << status << "\n";
		return 1;
	}
	
	for (size_t i = 0; i < N; i++) {
		if (cd[i] != ch[i]) {
			std::cout << "ERROR: Host and device results do not match (index " << i << ")!\n";
			return 2;
		}
	}
	
	std::cout << "SUCCESS!\n";	
	return 0;
}

 