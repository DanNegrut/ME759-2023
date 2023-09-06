#include <omp.h>
#include <stdio.h>

void main() {
    static long num_steps = 1000000;
    const double step = 1.0 / (double)num_steps;
    double sum = 0.0;
    double startT = omp_get_wtime();
    const int nThreadsUsed = 1;
#pragma omp parallel num_threads(nThreadsUsed)
    {
#pragma omp for simd reduction(+:sum) 
        for (int i = 0; i < num_steps; i++) {
            double x = (i + 0.5)*step;
            double dummy = 1. / (1. + x*x);
            sum += dummy;
        }
    }
    double myPi = 4.*step * sum;
    double timeElapsed = omp_get_wtime() - startT;

    printf("Pi is: %f\n", myPi);
    printf("It took this long [s]: %f\n", timeElapsed);
    printf("Used this many threads: %d\n", nThreadsUsed);
}
