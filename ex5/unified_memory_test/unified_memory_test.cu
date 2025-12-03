
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

__global__ void scale_kernel(float *data, float factor, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] *= factor;
    }
}

int main(int argc, char **argv) {
    // Adjust this to be *bigger* than your VRAM to force oversubscription.
    // Example: 6 GB allocation
    size_t bytes = (size_t)6 * 1024 * 1024 * 1024ULL; // 6 GiB

    if (argc > 1) {
        // Optional: allow size override in GiB: ./a.out 10  -> 10 GiB
        double gib = atof(argv[1]);
        bytes = (size_t)(gib * 1024.0 * 1024.0 * 1024.0);
    }

    size_t n = bytes / sizeof(float);
    printf("Allocating %.2f GiB (elements: %zu)\n",
           bytes / (1024.0 * 1024.0 * 1024.0), n);

    float *data = nullptr;

    // Unified Memory allocation
    CHECK_CUDA(cudaMallocManaged(&data, bytes));

    // Initialize on CPU (system RAM)
    for (size_t i = 0; i < n; ++i) {
        data[i] = 1.0f;
    }

    // Optionally: give driver a hint to prefer GPU
    int dev = 0;
    CHECK_CUDA(cudaGetDevice(&dev));
    // CHECK_CUDA(cudaMemAdvise(data, bytes, cudaMemAdviseSetPreferredLocation, dev));

    // Launch kernel on GPU – it will pull pages from system RAM as needed
    int threads = 256;
    int blocks  = (int)((n + threads - 1) / threads);

    printf("Launching kernel... this may be very slow once VRAM is exceeded.\n");
    scale_kernel<<<blocks, threads>>>(data, 2.0f, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Verify a few entries
    printf("data[0] = %f\n", data[0]);
    printf("data[n/2] = %f\n", data[n/2]);
    printf("data[n-1] = %f\n", data[n-1]);

    CHECK_CUDA(cudaFree(data));

    printf("Done.\n");
    return 0;
}
