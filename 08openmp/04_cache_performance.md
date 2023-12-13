# Cache Performance in Shared Memory 

The need for cache efficiency hasn't gone away just because we've started parallelising things; in fact, it may be more important than ever! Generally for distributed systems we just need to worry about the cache efficiency of each process in isolation, but if memory is shared then that means our cache gets shared too. The way that our cache behaves when shared is a little different though, so we'll need to re-think how we do things a bit. 

As always with the memory system, things will be system dependent, but on a typical CPU:
- General RAM and the largest cache level will be shared between cores which share memory i.e. there is just one physical RAM and one large physical cache (in my case just the L3 cache) which is accessed by all cores. (Not all cores necessarily access memory with equal bandwidth or latency though.) 
- Each core will have its own copy of the smallest cache level(s), in my case it's the L1 and L2 caches. 
- This keeps access to the small caches quick, but also enforces the need to consistency between the copies of the caches. 
    - If I have two cores $C_1$ and $C_2$ which both store a copy of variable `x` in their L1 cache, then when the want to read `x` from memory they just read it from the cache and not from RAM. Likewise when they want to _write_ to `x`, they write to the cache but not RAM. 
    - If $C_1$ changes `x`, it will change the value of `x` in _its own cache_, but not in the $C_2$ cache. 
    - In order for $C_2$ to read the correct value of `x` after the update, it has to find out about the change somehow. 
    - This mechanism will be system dependent but typically it will involve something written to a special shared area (possibly part of the L3 cache) when $C_1$ updates `x`. $C_2$ needs to check this and if `x` has been changed it needs to get the new value of `x` which will need to be copied over, incurring additional overheads.
- Remember that the cache stores data in blocks of a given size, called "cache lines". The cache lines on my machine for example are 64 bytes or 8 doubles wide. 
- If two L1 caches on different cores both store the same cache line, and one of the cores changes _any value in that line_, then **the entire cache line is invalidated for the other core**. 
    - This is extremely important as it means that even if the two cores never operate on the same values, if the values that they operate on are next to each other in memory and stored in the same cache line, then they will still invalidate one another's caches and cause lookups to have to be made. 

As a very simple example, let's look at a possible manual implementation of the reduction code shown in last week's class. I'll do this example with no compiler optimisations to prevent the compiler optimising away any memory read/write operations so we have full control! As a reminder, the basic code looks like this:

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include "timer.hpp"
#include <omp.h>

using namespace std;

int main() {
  const int num_threads = omp_get_max_threads();
  
  double sum = 0.;

  Timer timer;

#pragma omp parallel for 
  for(long i=0; i<N; ++i) {
    double x = i*dx;
    sum += 4.0f/(1.0f + x*x)*dx;
  }

  double elapsed = timer.elapsed();

  std::cout << "Time: " << elapsed << '\n';
  std::cout << "Result: " << sum << '\n';

  return 0;
}
```

- We know that this loop is both slow and inaccurate because of the data race problems discussed last week. 
- We can fix this using OMP's built in reductions. 
- Using 4 threads and the standard parallel sum reduction I get a time of 0.75 seconds for this sum. (My single threaded time is 2.05 seconds.)

Let's examine a simple reduction by hand though as an illustration of memory access patterns. We'll allocate some memory to hold each of our partial sums, and then get each thread to only interact with its own partial sum, and add them all together at the end. 

```cpp
int main() {
  const long N =   1'000'000'000;
  const double dx = 1.0f/(N-1);
  const int num_threads = omp_get_max_threads();

  double sum = 0.;
  
  double *partial_sums = new double[num_threads];

  Timer timer;

  #pragma omp parallel
  {
    int n = omp_get_thread_num();
    partial_sums[n] = 0.;

    #pragma omp for
    for(long i=0; i<N; ++i) {
      double x = i*dx;
      partial_sums[n] += 4.0f/(1.0f + x*x)*dx;
    }
  }

  for(int i = 0; i < num_threads; i++)
  {
    sum += partial_sums[i];
  }

  double elapsed = timer.elapsed();

  std::cout << "Time: " << elapsed << '\n';
  std::cout << "Result: " << sum << '\n';

  delete[] partial_sums;

  return 0;
}
```

- My result is 3.14159, which is correct, which suggests that we have solved our data race problem. 
- My time however is 2.10 seconds, so we've lost our performance boost! What is going on? 
- Each of our partial sums is next to one another in memory, and could even all fit in the same cache line since there are only 4 doubles. 
- Every time any thread updates its partial sum, any other thread whose partial sum is on the same cache line (which could be all of them!) will have its partial sum invalidated leading to a slower memory access. 
- This overhead is incurred at every single step in our loop for every thread! 

We can show that this is the problem by moving our variables to separate cache lines. We can force this to happen by buffering our partial sum array, since the array has to be contiguous. In my case, my cache is 64 bytes or 8 doubles wide, so we have to make sure there is an extra 7 doubles worth of space between each value in memory. 

```cpp
int main() {
  const long N =   1'000'000'000;
  const double dx = 1.0f/(N-1);
  const int num_threads = omp_get_max_threads();
  
  double sum = 0.;
  
  int LINE_SIZE = 8;
  double *partial_sums = new double[num_threads*8];

  Timer timer;

  #pragma omp parallel
  {
    int n = omp_get_thread_num()*8;
    partial_sums[n] = 0.;

    #pragma omp for
    for(long i=0; i<N; ++i) {
      double x = i*dx;
      partial_sums[n] += 4.0f/(1.0f + x*x)*dx;
    }
  }

  for(int i = 0; i < num_threads; i++)
  {
    sum += partial_sums[8*i];
  }

  double elapsed = timer.elapsed();

  std::cout << "Time: " << elapsed << '\n';
  std::cout << "Result: " << sum << '\n';

  delete[] partial_sums;

  return 0;
}
```

- By placing each partial sum 8 elements apart instead of 1 element apart, we force each value to exist on a different cache line. 
- Now all my partial sums can be in the cache at the same time _and_ updating them will not interfere with one another at all. 
- My time for this sum with four cores is 0.84s, almost back down to our OMP reduction time. 

The additional overhead is for the dereferencing operations, since we haven't compiled with optimisations. If for example we introduce a private variable to dereference the memory outside the for loop:

```cpp
  #pragma omp parallel
  {
    int n = omp_get_thread_num()*8;
    double &private_sum = partial_sums[n];
    private_sum = 0.;

    #pragma omp for
    for(long i=0; i<N; ++i) {
      double x = i*dx;
      private_sum += 4.0f/(1.0f + x*x)*dx;
    }
  }
```

then my time drops back down to 0.75 seconds. 

Don't worry too much if your threads access _some_ memory that lies next to memory accessed by another thread - most problems will have to divide up memory in a way that shares some kind of boundary! But if your threads are spending a lot of time accessing memory next to one another it can cause serious performance issues. Having thread work on contiguous blocks of data in an array for example is better than having threads work on data that is interleaved with other threads. 
