#include <stdio.h>
#include <cuda.h>

#define ITERATIONS 1000

#define CUDA_ERROR_CHECK(call) \
    do { \
        cudaError_t error = (call); \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDART error in %s (%s:%d):\n\t%s\n", __func__, __FILE__, __LINE__, cudaGetErrorString(error)); \
            return error; \
        } \
    } while (false)

/// "saxpy"
__global__ void saxpy(int n, float a, const float * __restrict__ x, int incx,
                      float * __restrict__ y, int incy) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] += a * x[i];
}

/// "main"
int main(int argc, char * argv[]) {
    int n = 10000, incx = 1, incy = 1;

    if (argc < 1 || argc > 4) {
        fprintf(stderr, "Usage: %s [n=10000 [incx=1 [incy=1]]]\n", argv[0]);
        return -1;
    }

    if (argc > 1) {
        if (sscanf(argv[1], "%d", &n) != 1) {
            fprintf(stderr, "Failed to parse n from '%s'\n", argv[1]);
            return -2;
        }

        if (argc > 2) {
            if (sscanf(argv[2], "%d", &incx) != 1) {
                fprintf(stderr, "Failed to parse incx from '%s'\n", argv[2]);
                return -2;
            }

            if (argc > 3) {
                if (sscanf(argv[3], "%d", &incy) != 1) {
                    fprintf(stderr, "Failed to parse incy from '%s'\n", argv[3]);
                    return -3;
                }
            }
        }
    }

    printf("n = %d, incx = %d, incy = %d\n", n, incx, incy);

    const float a = 1.5f;
    float * x, * y, * dx, * dy;
    CUDA_ERROR_CHECK(cudaMallocHost(&x, n * incx * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMallocHost(&y, n * incy * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMalloc(&dx, n * incx * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMalloc(&dy, n * incy * sizeof(float)));

    for (int i = 0; i < n; i++) {
        x[i * incx] = rand() / RAND_MAX;
        y[i * incy] = rand() / RAND_MAX;
    }
    
    CUDA_ERROR_CHECK(cudaMemcpy(dx, x, n * incx * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(dy, y, n * incy * sizeof(float), cudaMemcpyHostToDevice));

    float total = 0.0f;
    cudaEvent_t start, end;
    CUDA_ERROR_CHECK(cudaEventCreate(&start));
    CUDA_ERROR_CHECK(cudaEventCreate(&end));
    for (int i = 0; i < ITERATIONS; i++) {
        CUDA_ERROR_CHECK(cudaEventRecord(start, NULL));
        saxpy<<<n/64 + 1, 64>>>(n, a, dx, incx, dy, incy);
        CUDA_ERROR_CHECK(cudaEventRecord(end, NULL));
        CUDA_ERROR_CHECK(cudaEventSynchronize(end));
        float t;
        CUDA_ERROR_CHECK(cudaEventElapsedTime(&t, start, end));
        total += t;
    }
    CUDA_ERROR_CHECK(cudaEventDestroy(start));
    CUDA_ERROR_CHECK(cudaEventDestroy(end));
    printf("saxpy: %.6fms\n", total/ITERATIONS);

    CUDA_ERROR_CHECK(cudaFreeHost(x));
    CUDA_ERROR_CHECK(cudaFreeHost(y));
    CUDA_ERROR_CHECK(cudaFree(dx));
    CUDA_ERROR_CHECK(cudaFree(dy));

    return 0;
}
