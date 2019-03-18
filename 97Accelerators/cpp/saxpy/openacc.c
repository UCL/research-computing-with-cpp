#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cuda_runtime.h>

#define ITERATIONS 1000

#define CUDA_ERROR_CHECK(call) \
    do { \
        cudaError_t error = (call); \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s (%s:%d):\n\t%s\n", __func__, __FILE__, __LINE__, cudaGetErrorString(error)); \
            return error; \
        } \
    } while (false)

/// "saxpy"
void saxpy(int n, float a, const float * x, int incx, float restrict * y, int incy) {
#pragma acc kernels
  for (int i = 0; i < n; ++i)
    y[i * incy] = a * x[i * incx] + y[i * incy];
}

/// "main"
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

    const float a = 1.5f;
    float * x = malloc(n * incx * sizeof(float));
    float * y = malloc(n * incy * sizeof(float));

    if (x == NULL || y == NULL) {
        fputs("Failed to allocate x and y.", stderr);
        return -1;
    }

    for (int i = 0; i < n; i++) {
        x[i * incx] = rand() / RAND_MAX;
        y[i * incy] = rand() / RAND_MAX;
    }

    saxpy(n, a, x, incx, y, incy);

    // Perform the GPU SAXPY a lot of times and time the invocations
    float gpu_total = 0.0f;
    cudaEvent_t start, end;
    CUDA_ERROR_CHECK(cudaEventCreate(&start));
    CUDA_ERROR_CHECK(cudaEventCreate(&end));
    for (int i = 0; i < ITERATIONS; i++) {
        CUDA_ERROR_CHECK(cudaEventRecord(start, NULL));

        saxpy(n, a, x, incx, y, incy);

        CUDA_ERROR_CHECK(cudaEventRecord(end, NULL));
        CUDA_ERROR_CHECK(cudaEventSynchronize(end));
        float t;
        CUDA_ERROR_CHECK(cudaEventElapsedTime(&t, start, end));
        gpu_total += t;
    }
    CUDA_ERROR_CHECK(cudaEventDestroy(start));
    CUDA_ERROR_CHECK(cudaEventDestroy(end));

    free(x);
    free(y);

    // Print results
    double gpu_time = gpu_total / (1000 * ITERATIONS);
    double bandwidth = n * sizeof(float);
    double flops = 2 * n;
    printf("Bandwidth: %.3fGB/s\n", (bandwidth / gpu_time) / 1.e9);
    printf("Throughput: %.3fGFlops/s\n", (flops / gpu_time) / 1.e9);

    return 0;
}
