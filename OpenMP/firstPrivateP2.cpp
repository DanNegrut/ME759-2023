#include <omp.h>
#include <cstdio>

int main() {
  int i = 10;

#pragma omp parallel num_threads(4) firstprivate(i)
  {
    int threadID = omp_get_thread_num();
    printf("threadID = %d  i = %d\n", threadID, i);
    i = 1000 + threadID;
  }

  printf("i = %d\n", i);

  return 0;
}
