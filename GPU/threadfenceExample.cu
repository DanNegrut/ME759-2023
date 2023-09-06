#include<cuda.h>
#include<iostream>

__global__ void divergence(/* volatile */ int* data)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if ((i & 0x01) == 0)
    {
        data[i+1] = data[i+1] + i; // if even, come here
        //__threadfence_block();
    }
    else
    {
        data[i] = data[i] + 2*i; // if odd, come here
    }
}

int main()
{
    const int numElems = 4;
    int hostArray[numElems], *devArray;

    //allocate memory on the device (GPU); zero out all entries in this device array 
    cudaMalloc((void**)&devArray, sizeof(int) * numElems);
    cudaMemset(devArray, 0, numElems * sizeof(int));

    //invoke GPU kernel, with one block that has four threads
    divergence <<<1, numElems >>>(devArray);

    //bring the result back from the GPU into the hostArray 
    cudaMemcpy(&hostArray, devArray, sizeof(int) * numElems, cudaMemcpyDeviceToHost);

    //print out the result to confirm that things are looking good 
    std::cout << "Values stored in hostArray: " << std::endl;
    for (int i = 0; i < numElems; i++)
        std::cout << hostArray[i] << std::endl;

    //release the memory allocated on the GPU 
    cudaFree(devArray);
    return 0;
}
