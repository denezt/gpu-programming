#include <stdio.h>
#include <cuda_runtime.h>

__global__ void add(int a, int b, int *c) {
    *c = a + b;
}

// Error checking function
inline cudaError_t checkCudaErr(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), cudaGetErrorName(result), func);
        exit(EXIT_FAILURE);
    }
    return result;
}

#define CHECK_CUDA_ERR(val) checkCudaErr((val), #val, __FILE__, __LINE__)

int main(void) {
	int c;
	int *dev_c;
	for (int i=0; i < 10; i++){
		CHECK_CUDA_ERR(cudaMalloc((void**)&dev_c, sizeof(int)));
		add<<<1, 1>>>(2, 7, dev_c);
		CHECK_CUDA_ERR(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));
		printf("2 + 7 = %d\n", c);
		CHECK_CUDA_ERR(cudaFree(dev_c));
	}
	return 0;
}
