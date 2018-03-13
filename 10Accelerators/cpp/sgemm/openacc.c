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

/// "sgemm"
void sgemm(int m, int n, int k,
		   float a, const float * restrict A, int lda, const float * restrict B, int ldb,
		   float b, float restrict * C, int ldc) {
#pragma acc kernels
  for (int j = 0; j < n; j++) {
	  for (int i = 0; i < m; i++)
		  C[j * ldc + i] *= b;

	  for (int l = 0; l < k; l++) {
		  const float temp = a * B[j * ldb + k];
		  for (int i = 0; i < m; i++)
			  C[j * ldc + i] += temp * A[l * lda + i];
	  }
  }
}

/// "main"
int main(int argc, char * argv[]) {
    int m = 320, n = 640, k = 640;

    if (argc != 1 && argc != 4) {
        fprintf(stderr, "Usage: %s [m=320 n=640 k=640]\n", argv[0]);
        return -1;
    }

    if (argc == 4) {
		if (sscanf(argv[1], "%d", &m) != 1) {
			fprintf(stderr, "Failed to parse m from '%s'\n", argv[1]);
			return -2;
		}

		if (sscanf(argv[2], "%d", &n) != 1) {
			fprintf(stderr, "Failed to parse n from '%s'\n", argv[2]);
			return -3;
		}

		if (sscanf(argv[3], "%d", &k) != 1) {
			fprintf(stderr, "Failed to parse k from '%s'\n", argv[3]);
			return -4;
		}
    }

    printf("m = %d, n = %d, k = %d\n", m, n, k);

    float a = 0.5, * A, * B, b = 1.2, * C;

    // Round matrix column lengths up to multiple of SIMD width so each column
    // is correctly aligned in memory.  The value 3 is calculated as SIMD width /
    // sizeof(type) - 1 which for single precision floats using SSE is 128 / 32
    // - 1.  This does for host memory what cudaMallocPitch does for device memory.
    size_t lda = ((unsigned int)m + 3u) & ~3u;
    size_t ldb = ((unsigned int)k + 3u) & ~3u;
    size_t ldc = ((unsigned int)m + 3u) & ~3u;

    A = malloc(lda * k * sizeof(float));
    B = malloc(ldb * n * sizeof(float));
    C = malloc(ldc * n * sizeof(float));

    if (A == NULL || B == NULL || C == NULL) {
    	fputs("Failed to allocate matrices", stderr);
    	return -1;
    }

    // Initialise A, B and C ~U(0,1)
	for (size_t j = 0; j < k; j++) {
		for (size_t i = 0; i < m; i++)
    		A[j * lda + i] = (float)rand() / RAND_MAX;
    }
	for (size_t j = 0; j < n; j++) {
		for (size_t i = 0; i < k; i++)
    		B[j * ldb + i] = (float)rand() / RAND_MAX;
    }
	for (size_t j = 0; j < n; j++) {
		for (size_t i = 0; i < m; i++)
    		C[j * ldc + i] = (float)rand() / RAND_MAX;
    }

    sgemm(m, n, k, a, A, lda, B, ldb, b, C, ldc);

    // Perform the GPU SAXPY a lot of times and time the invocations
    float gpu_total = 0.0f;
    cudaEvent_t start, end;
    CUDA_ERROR_CHECK(cudaEventCreate(&start));
    CUDA_ERROR_CHECK(cudaEventCreate(&end));
    for (int i = 0; i < ITERATIONS; i++) {
        CUDA_ERROR_CHECK(cudaEventRecord(start, NULL));

        sgemm(m, n, k, a, A, lda, B, ldb, b, C, ldc);

        CUDA_ERROR_CHECK(cudaEventRecord(end, NULL));
        CUDA_ERROR_CHECK(cudaEventSynchronize(end));
        float t;
        CUDA_ERROR_CHECK(cudaEventElapsedTime(&t, start, end));
        gpu_total += t;
    }
    CUDA_ERROR_CHECK(cudaEventDestroy(start));
    CUDA_ERROR_CHECK(cudaEventDestroy(end));

    free(A);
    free(B);
    free(C);

    // Print results
    double gpu_time = gpu_total / (1000 * ITERATIONS);
    double bandwidth = n * sizeof(float);
    double flops = 2 * n;
    printf("Bandwidth: %.3fGB/s\n", (bandwidth / gpu_time) / 1.e9);
    printf("Throughput: %.3fGFlops/s\n", (flops / gpu_time) / 1.e9);

    return 0;
}
