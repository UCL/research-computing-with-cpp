---
title: Streaming
---

## Streaming computations

### *Single Instruction*, Multiple Data

All threads in a warp perform the exact same hardware instruction, but on different data.

On thread 0:

``` cpp
a[0] = alpha * x[0] + 1;
___syncthreads();
y[0] -= a[31];
```

On thread 31:

``` cpp
a[31] = alpha * x[0] + 1;
__syncthreads();
y[31] -= a[0];
```

### Is this single instruction?

``` cpp
if(thread_id < 16)
  a[thread_id] = alpha * x[thread_id] + 1;
else
  a[thread_id] = alpha * x[thread_id] - 1;

__syncthreads();
y[thread_id] -= a[31 - thread_id];
```

### Single Instruction, *Multiple Contiguous Data*

Each thread in a warp should access a consecutive address in memory.
Lets create a kernel for copying a vector using a single block consisting of a
single warp (of 32 threads).  Assume that the size of the vector is a multiple
of 32. `threadIdx.x` is the thread id in Cuda, and automagically passed to
each kernel.

The following is a jumble of too many expressions: put it back in order.

``` cpp
__global__ void copy(float *odata, const float *idata, int n)
{
  for (int j = 0; 32 * j < n; ++j)
  for (int j = 0; 32 * j < n; j += 32)
  for (int j = 0; j < n; ++j)
  for (int j = 0; j < n; j += 32)

  odata[j + threadIdx.x] = idata[j + threadIdx.x];
  odata[j + threadIdx.x * 32] = idata[j + threadIdx.x];
  odata[j + threadIdx.x] = idata[j + threadIdx.x * 32];
  odata[j + threadIdx.x * 32] = idata[j + threadIdx.x * 32];
}
```

### The joys of indexing

Now imagine rewriting the same code with n by m by p blocks of threads, where
each block is u by v by w threads. You have access to the size of the block
`blockDim.x`, the index and size of the grid (of blocks) `blockIdx.(x, y,
z)` and `gridDim.(x, y, z)`.

To limit memory transfers, each warp should read and write to contiguous arrays
in memory.

See also: [shared
memory](https://devblogs.nvidia.com/parallelforall/using-shared-memory-cuda-cc/),
[bank
conflicts](http://cuda-programming.blogspot.co.uk/2013/02/bank-conflicts-in-shared-memory-in-cuda.html)
