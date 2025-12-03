#include "../../common/book.h"
#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            return EXIT_FAILURE;                                             \
        }                                                                    \
    } while (0)

__global__ void increment(float *data, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] += 1.0f;
    }
}

int main() {
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("Using device %d: %s (sm_%d%d)\n", dev, prop.name, prop.major, prop.minor);

    const size_t n     = 1 << 24; // ~16M floats (~64 MB)
    const size_t bytes = n * sizeof(float);    
    float *host_ptr = nullptr;
    float *dev_ptr  = nullptr;

    // Allocate pinned, mapped host memory (system RAM)
    CHECK_CUDA(cudaHostAlloc(&host_ptr, bytes, cudaHostAllocMapped));
    // Get device pointer that maps to the same physical memory
    CHECK_CUDA(cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0));

    // Initialize in CPU RAM
    for (size_t i = 0; i < n; ++i) {
        host_ptr[i] = 42.0f;
    }

    // Launch kernel that *directly* operates on host memory via dev_ptr
    int threads = 256;
    int blocks  = (int)((n + threads - 1) / threads);
    increment<<<blocks, threads>>>(dev_ptr, n);
    // CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Check results on CPU
    printf("host_ptr[0]   = %f\n", host_ptr[0]);
    printf("host_ptr[n-1] = %f\n", host_ptr[n - 1]);

    CHECK_CUDA(cudaFreeHost(host_ptr));
    return 0;
}
