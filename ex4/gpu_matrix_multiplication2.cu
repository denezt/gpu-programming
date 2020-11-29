#include "../common/book.h"

__global__ void matrixMul(int *m, int *n, int *p, int size){
	// Calculate Row and Column
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int column = blockIdx.x * blockDim.x + threadIdx.x;

	// Partial Sum Element
	int p_sum;
	for (int i = 0; i < size; i++) {
		p_sum += m[row * size + i] * n[i * size + column];
		printf("psum: %i * %i\n",m[row * size + i], n[i * size + column]);
	}
	p[row * size + column] = p_sum;
}

int main() {
	int n = 1 << 10; // equals 1024 or 2^10
	printf("Square matrix of size %d\n", n);

	/**
	 * We should always start with our 
	 * host side of the code first
	 * and compile the code.
	 */

	// Host Matrix m,n,p
	int* h_m;	
	int* h_n;	
	int* h_p;	

	// Device Matrix m,n,p
	int* d_m;
	int* d_n;
	int* d_p;

	// Matrix Sizing n times n elements (integers)
	size_t bytes = n * n * sizeof(int); 

	// Allocate memory on host side
	h_m = (int*)malloc(bytes);
	h_n = (int*)malloc(bytes);
	h_p = (int*)malloc(bytes);

	// Initialize matrix m, n, p
	for(int i = 0; i < n; i++){
		for (int j = 0; j < n; j++){
			h_m[i*n + j] = rand() % 1024;
			h_n[i*n + j] = rand() % 1024;
		}
	}

	// Allocate memory on device side
	cudaMalloc(&d_m, bytes);
	cudaMalloc(&d_n, bytes);
	cudaMalloc(&d_p, bytes);

	// Copy data to the device
	cudaMemcpy(d_m, h_m, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_n, h_n, bytes, cudaMemcpyHostToDevice);
	
	int threads_per_block = 16;
	// Cuda specific parameter type (x,y,z) structs
	dim3 block_size(threads_per_block, threads_per_block);
	dim3 grid_size(n / block_size.x, n / block_size.y);

	printf("Grid size X: %d, Grid size Y: %d\n", grid_size.x, grid_size.y);

	matrixMul<<<grid_size,block_size>>>(d_m, d_n, d_p,n);

	// Copy to the host pointer from the device pointer
	cudaMemcpy(h_p, d_p, bytes, cudaMemcpyHostToDevice);



	return 0;
}
