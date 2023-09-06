// Author: Lijing Yang
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <omp.h>

using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::cout;

float reduce(const float* arr, const size_t m, const size_t n) {
    float sum = 0.0;
    #pragma omp parallel for collapse(2) reduction(+:sum)
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            sum += arr[i * n + j];
        }
    }
    return sum;
}

int main(int argc, char* argv[]) {
    size_t m = atoi(argv[1]);
    size_t n = atoi(argv[2]);
    size_t t = atoi(argv[3]);
    float *arr = (float*) malloc(m*n*sizeof(float));

    #pragma omp parallel for
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            arr[i * n + j] = 1.1f;
        }
    }
    
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    float res;
    omp_set_num_threads(t);
    start = high_resolution_clock::now();
    for (size_t i = 0; i < 10; i++) {
        res = reduce(arr, m, n);
    }
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>> (end - start) / 10;
   
    cout << "Result: " << res << "\n";
    cout << t << " " << duration_sec.count() << "\n";
    
    free(arr);   
	return 0;
}

