#define _CRT_SECURE_NO_DEPRECATE
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <vector>

// Read in vector of pixel vales
std::vector<int> *read_image(FILE *fp, int d1, int d2) {
    int xVal;
    int len = d1 * d2;
    std::vector<int> *x = new std::vector<int>(len);
    for (int i = 0; i < len; i++) {
        fscanf(fp, "%d ", &xVal);
        (*x)[i] = xVal;
    }
    return x;
}

int main(int argc, char *argv[]) {
    // Set number of threads
    const auto num_threads = std::atoi(argv[1]);
    omp_set_num_threads(num_threads);

    // Read in Image
    FILE *fid = std::fopen("picture.inp", "r");
    int d1 = 1800;
    int d2 = 1200;
    int len = d1 * d2;
    std::vector<int> *image = read_image(fid, d1, d2);

    // Histogram Counter
    std::vector<int> histCounter = std::vector<int>(7);

    // Declaration of the std:vector addition
#pragma omp declare reduction(vec_int_add : std::vector<int> : \
								std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<int>())) \
								initializer(omp_priv = omp_orig)

    // Reset Histogram Counter to Zero
    std::fill(histCounter.begin(), histCounter.end(), 0);

    double startTime = omp_get_wtime(); // Start Clock

#pragma omp parallel for reduction(vec_int_add : histCounter)
                                        // Perfrom the histrogram; shared(histCounter)
    for (int jj = 0; jj < len; jj++)
        (histCounter)[(*image)[jj]]++;

    double endTime = omp_get_wtime(); // Stop Clock

                                      // Ouput information
    for (int ii = 0; ii < 7; ii++)
        std::cout << (histCounter)[ii] << std::endl;
    std::cout << num_threads << std::endl;
    std::cout << std::setprecision(16) << (endTime - startTime) * 1000 << std::endl;

    // delete(histCounter);
    delete (image);
    return 0;
}
