#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/generate.h>
#include <thrust/functional.h>
#include <cuda_runtime.h>
#include <iostream>

#define ITERATIONS 1000

/// "sgemm"
struct saxpy_functor : public thrust::binary_function<float, float, float> {
    
    const float a;
    
    saxpy_functor(float _a) : a(_a) {}
    
    __host__ __device__ float operator()(const float &x, const float &y) const {
        return a * x + y;
    }
};

void saxpy(const float & a, const thrust::device_vector<float> & x, thrust::device_vector<float> & y) {
    thrust::transform(x.begin(), x.end(), y.begin(), y.begin(), saxpy_functor(a));
}

/// "rand01"
__host__ float rand01() {
    return std::rand() / RAND_MAX;
}

int main(void) {
    const int n = 1048576;
    const float a = rand01();
    thrust::host_vector<float> x(n);
    thrust::host_vector<float> y(n);

    thrust::generate(x.begin(), x.end(), rand01);
    thrust::generate(y.begin(), y.end(), rand01);

    thrust::device_vector<float> dx(x);
    thrust::device_vector<float> dy(y);
    
    saxpy(a, dx, dy);
    
    float gpu_total = 0.0f;
    cudaEvent_t start, end;

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    for (int i = 0; i < ITERATIONS; i++) {
        cudaEventRecord(start, NULL);

        saxpy(a, dx, dy);

        cudaEventRecord(end, NULL);
        cudaEventSynchronize(end);
        float t;
        cudaEventElapsedTime(&t, start, end);
        gpu_total += t;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    // Print results
    double gpu_time = gpu_total / (1000 * ITERATIONS);
    double bandwidth = n * sizeof(float);
    double flops = 2 * n;
    std::cout << "Bandwidth: " << (bandwidth / gpu_time) / 1.e9 << "GB/s" << std::endl
    std::cout << "Throughput: " << (flops / gpu_time) / 1.e9 << "GFlops/s" << std::endl;

    return 0;
}
