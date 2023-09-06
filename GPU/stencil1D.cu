#include <stdlib.h>
#include "cuda.h"
#include <iostream>

#define RADIUS 3
#define N 1000000

void initializeWeights(float* weights) {
    weights[0] = 0.05f;
    weights[1] = 0.10f;
    weights[2] = 0.20f;
    weights[3] = 0.30f;
    weights[4] = 0.20f;
    weights[5] = 0.10f;
    weights[6] = 0.05f;
}

void initializeArray(float* in) {
    for (int i = 0; i < N; i++) {
        in[i] = rand() % 10;
    }
}

__global__ void applyStencil1D(int sIdx, int eIdx, const float* weights, float* in, float* out) {
    int i = sIdx + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < eIdx) {
        out[i] = 0;
        //loop over all elements in the stencil
        for (int j = -RADIUS; j <= RADIUS; j++) {
            out[i] += weights[j + RADIUS] * in[i + j];
        }
        out[i] = out[i] / (2 * RADIUS + 1);
    }
}

int main() {
    int wsize = 2 * RADIUS + 1;
    //allocate resources
    float* weights = new float[wsize];
    float* in = new float[N];
    float* out = new float[N];
    initializeWeights(weights);
    initializeArray(in);

    float* d_weights;
    cudaMalloc(&d_weights, wsize * sizeof(float));
    
    float* d_in;
    cudaMalloc(&d_in, N * sizeof(float));
    
    float* d_out;
    cudaMalloc(&d_out, N * sizeof(float));

    cudaMemcpy(d_weights, weights, wsize*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in, in, N*sizeof(float), cudaMemcpyHostToDevice);

    applyStencil1D <<<(N+511) / 512, 512 >>> (RADIUS, N - RADIUS, d_weights, d_in, d_out);

    cudaMemcpy(out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);

    //free resources
    delete[] weights;
    delete[] in;
    delete[] out;

    cudaFree(d_weights);
    cudaFree(d_in);
    cudaFree(d_out);
}
