#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char ** argv)
{
    #pragma omp parallel
    {   
        int threadnum = 0;
        int numthreads = 0;
        #ifdef _OPENMP
            threadnum = omp_get_thread_num();
            numthreads = omp_get_num_threads();
        #endif
        #pragma omp critical
        {
            std::cout << "Hello World, I am " << threadnum << " of " << numthreads << std::endl;
        }
    }
}