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

    CUDA_ERROR_CHECK(cudaMallocHost((void **)&A, lda * k * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMallocHost((void **)&B, ldb * n * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMallocHost((void **)&C, ldc * n * sizeof(float)));

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

	/// "cublas_sgemm"
	// Allocate matrices on GPU
    float * dA, * dB, * dC;
    size_t dlda, dldb, dldc;
    CUDA_ERROR_CHECK(cudaMallocPitch((void **)&dA, &dlda, m * sizeof(float), k));
    CUDA_ERROR_CHECK(cudaMallocPitch((void **)&dB, &dldb, k * sizeof(float), n));
    CUDA_ERROR_CHECK(cudaMallocPitch((void **)&dC, &dldc, m * sizeof(float), n));

    // cudaMallocPitch returns leading dimensions in bytes while CUBLAS expects
    // them as number of elements
    dlda /= sizeof(float);
    dldb /= sizeof(float);
    dldc /= sizeof(float);

    // Create CUBLAS handle
    cublasHandle_t handle;
    CUBLAS_ERROR_CHECK(cublasCreate(&handle));

    // Copy matrices into GPU memory
    CUBLAS_ERROR_CHECK(cublasSetMatrix(m, k, sizeof(float), A, lda, dA, dlda));
    CUBLAS_ERROR_CHECK(cublasSetMatrix(k, n, sizeof(float), B, ldb, dB, dldb));
    CUBLAS_ERROR_CHECK(cublasSetMatrix(m, n, sizeof(float), C, ldc, dC, dldc));

    // Perform the GPU SGEMM
    CUBLAS_ERROR_CHECK(cublasSgemm(handle,
    		                       CUBLAS_OP_N, CUBLAS_OP_N,
								   m, n, k,
								   &a, dA, dlda, dB, dldb,
								   &b, dC, dldc));

    // Copy results back into CPU memory
    CUBLAS_ERROR_CHECK(cublasGetMatrix(m, n, sizeof(float), dC, dldc, C, ldc));

    /// "benchmark"
    // Perform the GPU SGEMM a lot of times and time the invocations
    float gpu_total = 0.0f;
    cudaEvent_t start, end;
    CUDA_ERROR_CHECK(cudaEventCreate(&start));
    CUDA_ERROR_CHECK(cudaEventCreate(&end));
    for (int i = 0; i < ITERATIONS; i++) {
        CUDA_ERROR_CHECK(cudaEventRecord(start, NULL));

        CUBLAS_ERROR_CHECK(cublasSgemm(handle,
        		                       CUBLAS_OP_N, CUBLAS_OP_N,
									   m, n, k,
									   &a, dA, dlda, dB, dldb,
									   &b, dC, dldc));

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
    CUDA_ERROR_CHECK(cudaFree(dA));
    CUDA_ERROR_CHECK(cudaFree(dB));
    CUDA_ERROR_CHECK(cudaFree(dC));
    CUDA_ERROR_CHECK(cudaFreeHost(A));
    CUDA_ERROR_CHECK(cudaFreeHost(B));
    CUDA_ERROR_CHECK(cudaFreeHost(C));

    // Print results
    double gpu_time = gpu_total / (1000 * ITERATIONS);
    double bandwidth = 2 * m * n * k * sizeof(float);
    double flops = 2 * m * n * k;
    printf("Bandwidth: %.3fGB/s\n", (bandwidth / gpu_time) / 1.e9);
    printf("Throughput: %.3fGFlops/s\n", (flops / gpu_time) / 1.e9);

    return 0;
}
