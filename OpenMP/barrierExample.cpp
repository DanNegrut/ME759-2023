#include <omp.h>
#include <iostream>

void foo(int *fooVal)
{
    *fooVal = 10;
#pragma omp barrier
}

void bar(int *barVal)
{
#pragma omp barrier
    *barVal = -10;
}

int main()
{
    int fVal = -1;
    int bVal = 1;

    std::cout << "fVal and bVal before: " << fVal << " and " << bVal << std::endl;

    omp_set_num_threads(2);

#pragma omp parallel
    {
        int threadID = omp_get_thread_num();

        if (threadID == 0)
            foo(&fVal);
        else if (threadID == 1)
            bar(&bVal);
        else
        {
            std::cout << "OpenMP choked" << std::endl;
        }
    }

    std::cout << "fVal and bVal after: " << fVal << " and " << bVal << std::endl;

    return 0;
}