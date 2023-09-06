
#define EXPORT_KERNEL(k) extern "C" void* kern_##k = (void*)k


// Super simple vector kernel	
extern "C" __global__ void add_vector(int* a, int* b, int* c) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	c[i] = a[i] + b[i];
}
EXPORT_KERNEL(add_vector);