// An intermediate level GPU Program

#include <stdio.h>
#include <stdlib.h>
#include "../common/book.h"


__global__ void vectorAdd(int *a, int *b, int *c, int n){
	// Calculate Index Thread
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// Make sure we stay in-bounds
	if (tid < n){
		// Vector Add
		c[tid] = a[tid] + b[tid];
	}
}

int main(){
	// Number of elements
	int n = 1 << 20;

	// Host Pointers
	int *h_a;
	int *h_b;
	int *h_c;

	// Device Pointers
	int *d_a;
	int *d_b;
	int *d_c;

	size_t bytes = n * sizeof(int);

	// Allocate memory on host side
	h_a = (int*)malloc(bytes);
	h_b = (int*)malloc(bytes);
	h_c = (int*)malloc(bytes);

	for (int i = 0; i < n; i++){
		h_a[i] = 1;
		h_b[i] = 2;
	}

	// Allocate memory on device side
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);
	
	// Init block and grid size
	int block_size = 1024;
	int grid_size = (int)ceil((float) n /block_size);
	printf("Grid size is %d\n", grid_size);

	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

	vectorAdd<<<grid_size, block_size>>>(d_a, d_b, d_c, n);

	cudaMemcpy(h_c,d_c,bytes, cudaMemcpyHostToDevice);

	for(int i =0; i < n; i++){
		if (h_c[i] != 3){
			printf("ERROR!\n");
			break;
		}
	}

	printf("Completed Successfully!\n");

	free(h_a);
	free(h_b);
	free(h_c);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return 0;
}
