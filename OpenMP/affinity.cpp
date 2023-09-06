#include <omp.h>
#include <stdio.h>


int main() {
    int reps = 2;
    int N = 6;
    int a = 0;

    double start = omp_get_wtime();
#pragma omp parallel
    { // this part done by all threads, in parallel
#pragma omp single
        printf("Number of threads: %d\n", omp_get_num_threads());

        for (int j = 0; j < reps; j++) {
#pragma omp for schedule(static,1)
            for (int i = 0; i < N; i++) {
#pragma omp atomic
                a++;
            }

        }
    }
    double end = omp_get_wtime();

    printf("Work took: %f seconds\n", end - start);
    printf("Result: a = %d\n", a);
    return 0;
}