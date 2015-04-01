/*
 * CUDA SGEMM implementation.
 * Copied from https://github.com/garymacindoe/cuda-cholesky/blob/master/blas/sgemm.cu
 * 
 * Author: Gary Macindoe
 * Date: 02/05/2013
 * Modified for inclusion in Research Computing with C++ 31/03/2015
 */
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

typedef enum { CBlasNoTrans = 'N', CBlasTrans = 'T', CBlasConjTrans = 'C' } CBlasTranspose;

/// "saxpy"
// y(1:16) += alpha * x(1:16)
__device__ void saxpy(float alpha, const float * __restrict__ x, float * __restrict__ y) {
  y[ 0] += alpha * x[ 0]; y[ 1] += alpha * x[ 1]; y[ 2] += alpha * x[ 2]; y[ 3] += alpha * x[ 3];
  y[ 4] += alpha * x[ 4]; y[ 5] += alpha * x[ 5]; y[ 6] += alpha * x[ 6]; y[ 7] += alpha * x[ 7];
  y[ 8] += alpha * x[ 8]; y[ 9] += alpha * x[ 9]; y[10] += alpha * x[10]; y[11] += alpha * x[11];
  y[12] += alpha * x[12]; y[13] += alpha * x[13]; y[14] += alpha * x[14]; y[15] += alpha * x[15];
}

/**
 * SGEMM:
 *   C := alpha * AB   + beta * C for transB == CBlasNoTrans
 *   C := alpha * AB'  + beta * C for transB == CBlasTrans
 *
 * @param transB  transpose for B.
 * @param mb      the number of rows in the block of C.
 * @param nb      the number of columns in the block of C.
 * @param kb      how far to unroll the inner loop.
 * @param bx      blockDim.x.
 * @param by      blockDim.y.
 */
template <CBlasTranspose transB,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void sgemm(const float * __restrict__ A, const float * __restrict__ B,
                      float * __restrict__ C,
                      float alpha, float beta,
                      int lda, int ldb, int ldc,
                      int m, int n, int k) {

  const int bi = blockIdx.x * mb;       // Starting row of block of C/C
  const int bj = blockIdx.y * nb;       // Starting column of block of C/C
  int ti = threadIdx.y * bx + threadIdx.x;
  int tj = 0;

  /*
   * Compute our starting points in A, B, C and C.
   *
   * For transA != CBlasNoTrans A is cached in shared memory so the unwrapped
   * thread index can be re-wrapped around mb when calculating C.
   *
   * If transA == CBlasNoTrans then bx * by == mb (checked later on) so there
   * doesn't need to be a separate check for transA == CBlasNoTrans in
   * calculating the start of C/C here.
   */
  A += bi + ti;
  B += (transB == CBlasNoTrans) ? (bj + threadIdx.y) * ldb + threadIdx.x : threadIdx.y * ldb + bj + threadIdx.x;
  C += (bj + tj) * ldc + bi + ti;
  n -= bj + tj;
  m -= bi + ti;

  /*
   * Block B in shared memory and C in registers.
   */
  __shared__ float b[kb][(transB == CBlasNoTrans) ? nb + 1 : nb];

  float c[] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

  while (k > 0) {
    // B will always be "transposed" w.r.t. C so must always be cached in shared
    // memory (i.e. it is read along the K or N dimensions when M is the
    // dimension being expanded).
    if (transB == CBlasNoTrans) {
#pragma unroll
      for (int j = 0; j < nb; j += by)
        b[threadIdx.x][j + threadIdx.y] = B[j * ldb];
    }
    else {
#pragma unroll
      for (int l = 0; l < kb; l += by) {
#pragma unroll
        for (int j = 0; j < nb; j += bx)
          b[l + threadIdx.y][j + threadIdx.x] = B[l * ldb + j];
      }
    }

    __syncthreads();

    if (k < kb) break;

    // Read A straight from global memory.
#pragma unroll
    for (int l = 0; l < kb; l++) {
      saxpy(A[0], b[l], c);
      A += lda;
    }

    __syncthreads();

    B += (transB == CBlasNoTrans) ? kb : kb * ldb;
    k -= kb;
  }

  for (int l = 0; l < k; l++) {
    saxpy(A[0], b[l], c);
    A += lda;
  }

  if (n <= 0 || m <= 0) return;
  if (beta == 0.0f) {
    C[0] = alpha * c[ 0]; if ( 1 >= n) return; C += ldc;
    C[0] = alpha * c[ 1]; if ( 2 >= n) return; C += ldc;
    C[0] = alpha * c[ 2]; if ( 3 >= n) return; C += ldc;
    C[0] = alpha * c[ 3]; if ( 4 >= n) return; C += ldc;
    C[0] = alpha * c[ 4]; if ( 5 >= n) return; C += ldc;
    C[0] = alpha * c[ 5]; if ( 6 >= n) return; C += ldc;
    C[0] = alpha * c[ 6]; if ( 7 >= n) return; C += ldc;
    C[0] = alpha * c[ 7]; if ( 8 >= n) return; C += ldc;
    C[0] = alpha * c[ 8]; if ( 9 >= n) return; C += ldc;
    C[0] = alpha * c[ 9]; if (10 >= n) return; C += ldc;
    C[0] = alpha * c[10]; if (11 >= n) return; C += ldc;
    C[0] = alpha * c[11]; if (12 >= n) return; C += ldc;
    C[0] = alpha * c[12]; if (13 >= n) return; C += ldc;
    C[0] = alpha * c[13]; if (14 >= n) return; C += ldc;
    C[0] = alpha * c[14]; if (15 >= n) return; C += ldc;
    C[0] = alpha * c[15];
  }
  else {
    C[0] = alpha * c[ 0] + beta * C[0]; if ( 1 >= n) return; C += ldc;
    C[0] = alpha * c[ 1] + beta * C[0]; if ( 2 >= n) return; C += ldc;
    C[0] = alpha * c[ 2] + beta * C[0]; if ( 3 >= n) return; C += ldc;
    C[0] = alpha * c[ 3] + beta * C[0]; if ( 4 >= n) return; C += ldc;
    C[0] = alpha * c[ 4] + beta * C[0]; if ( 5 >= n) return; C += ldc;
    C[0] = alpha * c[ 5] + beta * C[0]; if ( 6 >= n) return; C += ldc;
    C[0] = alpha * c[ 6] + beta * C[0]; if ( 7 >= n) return; C += ldc;
    C[0] = alpha * c[ 7] + beta * C[0]; if ( 8 >= n) return; C += ldc;
    C[0] = alpha * c[ 8] + beta * C[0]; if ( 9 >= n) return; C += ldc;
    C[0] = alpha * c[ 9] + beta * C[0]; if (10 >= n) return; C += ldc;
    C[0] = alpha * c[10] + beta * C[0]; if (11 >= n) return; C += ldc;
    C[0] = alpha * c[11] + beta * C[0]; if (12 >= n) return; C += ldc;
    C[0] = alpha * c[12] + beta * C[0]; if (13 >= n) return; C += ldc;
    C[0] = alpha * c[13] + beta * C[0]; if (14 >= n) return; C += ldc;
    C[0] = alpha * c[14] + beta * C[0]; if (15 >= n) return; C += ldc;
    C[0] = alpha * c[15] + beta * C[0];
  }
}

/**
 * For C = aAB + bC:
 *   mb must be a multiple of the warp size (32) and less than or equal to the
 *        maximum number of threads per block (512).
 *   nb must be less than or equal to 20 (registers start spilling to global
 *        memory after 20).
 *   kb must be a multiple of the half-warp size (16) and such that
 *        (nb + 1)*kb*sizeof(float) is less than the amount of shared memory
 *        available per block (16384 bytes).
 *
 * mb and nb must be selected such that the bandwidth reduction is greater than
 * the flop:word ratio of the GPU.  The bandwidth reduction for all valid values
 * of mb and nb can be calculated with the following loop (bash):
 * echo -n " mb\nb"; for nb in {1..20}; do printf "%6d" ${nb}; done; echo; for mb in {32..512..32}; do printf "%6d"  ${mb}; for nb in {1..20}; do printf "%6.2f" $(echo 2 / \(1/${mb} + 1/${nb}\) | bc -l); done; echo; done
 *
 * Sample output:
 *  mb\nb     1     2     3     4     5     6     7     8     9    10    11    12    13    14    15    16    17    18    19    20
 *     32  1.94  3.76  5.49  7.11  8.65 10.11 11.49 12.80 14.05 15.24 16.37 17.45 18.49 19.48 20.43 21.33 22.20 23.04 23.84 24.62
 *     64  1.97  3.88  5.73  7.53  9.28 10.97 12.62 14.22 15.78 17.30 18.77 20.21 21.61 22.97 24.30 25.60 26.86 28.10 29.30 30.48
 *     96  1.98  3.92  5.82  7.68  9.50 11.29 13.05 14.77 16.46 18.11 19.74 21.33 22.90 24.44 25.95 27.43 28.88 30.32 31.72 33.10
 *    128  1.98  3.94  5.86  7.76  9.62 11.46 13.27 15.06 16.82 18.55 20.26 21.94 23.60 25.24 26.85 28.44 30.01 31.56 33.09 34.59
 *    160  1.99  3.95  5.89  7.80  9.70 11.57 13.41 15.24 17.04 18.82 20.58 22.33 24.05 25.75 27.43 29.09 30.73 32.36 33.97 35.56
 *    192  1.99  3.96  5.91  7.84  9.75 11.64 13.51 15.36 17.19 19.01 20.81 22.59 24.35 26.10 27.83 29.54 31.23 32.91 34.58 36.23
 *    224  1.99  3.96  5.92  7.86  9.78 11.69 13.58 15.45 17.30 19.15 20.97 22.78 24.57 26.35 28.12 29.87 31.60 33.32 35.03 36.72
 *    256  1.99  3.97  5.93  7.88  9.81 11.73 13.63 15.52 17.39 19.25 21.09 22.93 24.74 26.55 28.34 30.12 31.88 33.64 35.37 37.10
 *    288  1.99  3.97  5.94  7.89  9.83 11.76 13.67 15.57 17.45 19.33 21.19 23.04 24.88 26.70 28.51 30.32 32.10 33.88 35.65 37.40
 *    320  1.99  3.98  5.94  7.90  9.85 11.78 13.70 15.61 17.51 19.39 21.27 23.13 24.98 26.83 28.66 30.48 32.28 34.08 35.87 37.65
 *    352  1.99  3.98  5.95  7.91  9.86 11.80 13.73 15.64 17.55 19.45 21.33 23.21 25.07 26.93 28.77 30.61 32.43 34.25 36.05 37.85
 *    384  1.99  3.98  5.95  7.92  9.87 11.82 13.75 15.67 17.59 19.49 21.39 23.27 25.15 27.02 28.87 30.72 32.56 34.39 36.21 38.02
 *    416  2.00  3.98  5.96  7.92  9.88 11.83 13.77 15.70 17.62 19.53 21.43 23.33 25.21 27.09 28.96 30.81 32.67 34.51 36.34 38.17
 *    448  2.00  3.98  5.96  7.93  9.89 11.84 13.78 15.72 17.65 19.56 21.47 23.37 25.27 27.15 29.03 30.90 32.76 34.61 36.45 38.29
 *    480  2.00  3.98  5.96  7.93  9.90 11.85 13.80 15.74 17.67 19.59 21.51 23.41 25.31 27.21 29.09 30.97 32.84 34.70 36.55 38.40
 *    512  2.00  3.98  5.97  7.94  9.90 11.86 13.81 15.75 17.69 19.62 21.54 23.45 25.36 27.25 29.15 31.03 32.91 34.78 36.64 38.50
 *
 * The number of registers per block is mb*32 (compiled with -maxrregcount=32).
 * More threads == better performance (from flop-test) therefore mb is chosen to
 * be the largest number of threads such that the number of blocks per
 * multiprocessor is still limited by the register usage.
 * kb is chosen to be the largest multiple of 16 such that the number of blocks
 * per multiprocessor is limited by the register usage.
 */

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

    CUDA_ERROR_CHECK(cudaMallocHost(&A, lda * k * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMallocHost(&B, ldb * n * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMallocHost(&C, ldc * n * sizeof(float)));

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

    // cudaMallocPitch returns leading dimensions in bytes while SGEMM expects
    // them as number of elements
    dlda /= sizeof(float);
    dldb /= sizeof(float);
    dldc /= sizeof(float);

    CUDA_ERROR_CHECK(cudaMemcpy2D(dA, dlda * sizeof(float), A, lda * sizeof(float), m * sizeof(float), k, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy2D(dB, dldb * sizeof(float), B, ldb * sizeof(float), k * sizeof(float), n, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy2D(dC, dldc * sizeof(float), C, ldc * sizeof(float), m * sizeof(float), n, cudaMemcpyHostToDevice));

    float gpu_total = 0.0f;
    cudaEvent_t start, end;
    CUDA_ERROR_CHECK(cudaEventCreate(&start));
    CUDA_ERROR_CHECK(cudaEventCreate(&end));
    for (int i = 0; i < ITERATIONS; i++) {
        CUDA_ERROR_CHECK(cudaEventRecord(start, NULL));

        dim3 blocks((m + 63) / 64, (n + 15) / 16);
        dim3 threads(16, 16);
        sgemm<CBlasNoTrans, 64, 16, 16, 16,  4><<<blocks, threads>>>(dA, dB, dC, a, b, dlda, dldb, dldc, m, n, k);

        CUDA_ERROR_CHECK(cudaEventRecord(end, NULL));
        CUDA_ERROR_CHECK(cudaEventSynchronize(end));
        float t;
        CUDA_ERROR_CHECK(cudaEventElapsedTime(&t, start, end));
        gpu_total += t;
    }
    CUDA_ERROR_CHECK(cudaEventDestroy(start));
    CUDA_ERROR_CHECK(cudaEventDestroy(end));

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
