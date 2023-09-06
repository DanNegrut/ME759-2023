#include <omp.h>
#include <stdio.h>

void simple(int n, float *a, float *b) {
    int i;

#pragma omp parallel 
    {
#pragma omp single
        printf("Number of threads: %d\n", omp_get_num_threads());
#pragma omp for
        for (i = 1; i < n; i++) /* i is private by default */
            b[i] = (a[i] + a[i - 1]) / 2.0f;
    }
}

int main() {
    const size_t N = 4;
    float a[N] = { 1.f, 2.f, 3.f, -4.f };
    float b[N] = { -10.f, -10.f, -10.f, -10.f };
    simple(N, a, b);
    for (int i = 0; i < N; i++)
        printf("%d: %f\n", i, b[i]);
    return 0;
}