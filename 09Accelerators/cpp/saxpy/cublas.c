#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define ITERATIONS 1000

#define CUDA_ERROR_CHECK(call) \
    do { \
        cudaError_t error = (call); \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s (%s:%d):\n\t%s\n", __func__, __FILE__, __LINE__, cudaGetErrorString(error)); \
            return error; \
        } \
    } while (false)

#define CUBLAS_ERROR_CHECK(call) \
    do { \
        cublasStatus_t status = (call); \
        if (status != cudaSuccess) { \
            fprintf(stderr, "CUBLAS error in %s (%s:%d)\n", __func__, __FILE__, __LINE__); \
            return status; \
        } \
    } while (false)

int main(int argc, char * argv[]) {
    int n = 1048576, incx = 1, incy = 1;

    if (argc < 1 || argc > 4) {
        fprintf(stderr, "Usage: %s [n=%d [incx=%d [incy=%d]]]\n", argv[0], n, incx, incy);
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
				return -3;
			}

			if (argc > 3) {
				if (sscanf(argv[3], "%d", &incy) != 1) {
					fprintf(stderr, "Failed to parse incy from '%s'\n", argv[3]);
					return -4;
				}
			}
		}
    }

    printf("n = %d, incx = %d, incy = %d\n", n, incx, incy);

    // Allocate x and y in pinned (non-pageable) memory
    const float a = 1.5f;
    float * x, * y;
    CUDA_ERROR_CHECK(cudaMallocHost((void **)&x, n * incx * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMallocHost((void **)&y, n * incy * sizeof(float)));

    // Initialise x and y
    for (int i = 0; i < n; i++) {
        x[i * incx] = rand() / RAND_MAX;
        y[i * incy] = rand() / RAND_MAX;
    }

    /// "cublas_saxpy"
	// Allocate vectors on GPU
    float * dx, * dy;
    CUDA_ERROR_CHECK(cudaMalloc((void **)&dx, n * incx * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMalloc((void **)&dy, n * incy * sizeof(float)));

    // Create CUBLAS handle
    cublasHandle_t handle;
    CUBLAS_ERROR_CHECK(cublasCreate(&handle));

    // Copy vectors into GPU memory
    CUBLAS_ERROR_CHECK(cublasSetVector(n, sizeof(float), x, incx, dx, incx));
    CUBLAS_ERROR_CHECK(cublasSetVector(n, sizeof(float), y, incy, dy, incy));

    // Perform the GPU SAXPY
    CUBLAS_ERROR_CHECK(cublasSaxpy(handle, n, &a, dx, incx, dy, incy));

    // Copy results back into CPU memory
    CUBLAS_ERROR_CHECK(cublasGetVector(n, sizeof(float), dy, incy, y, incy));

    /// "benchmark"
    // Perform the GPU SAXPY a lot of times and time the invocations
    float gpu_total = 0.0f;
    cudaEvent_t start, end;
    CUDA_ERROR_CHECK(cudaEventCreate(&start));
    CUDA_ERROR_CHECK(cudaEventCreate(&end));
    for (int i = 0; i < ITERATIONS; i++) {
        CUDA_ERROR_CHECK(cudaEventRecord(start, NULL));

        CUBLAS_ERROR_CHECK(cublasSaxpy(handle, n, &a, dx, incx, dy, incy));

        CUDA_ERROR_CHECK(cudaEventRecord(end, NULL));
        CUDA_ERROR_CHECK(cudaEventSynchronize(end));
        float t;
        CUDA_ERROR_CHECK(cudaEventElapsedTime(&t, start, end));
        gpu_total += t;
    }
    CUDA_ERROR_CHECK(cudaEventDestroy(start));
    CUDA_ERROR_CHECK(cudaEventDestroy(end));

    // Destroy CUBLAS handle
    CUBLAS_ERROR_CHECK(cublasDestroy(handle));

    // Free memory
    CUDA_ERROR_CHECK(cudaFree(dx));
    CUDA_ERROR_CHECK(cudaFree(dy));
    CUDA_ERROR_CHECK(cudaFreeHost(x));
    CUDA_ERROR_CHECK(cudaFreeHost(y));

    // Print results
    double gpu_time = gpu_total / (1000 * ITERATIONS);
    double bandwidth = n * sizeof(float);
    double flops = 2 * n;
    printf("Bandwidth: %.3fGB/s\n", (bandwidth / gpu_time) / 1.e9);
    printf("Throughput: %.3fGFlops/s\n", (flops / gpu_time) / 1.e9);

    return 0;
}
