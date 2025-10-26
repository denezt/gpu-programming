#include "stdio.h"
#include "stdlib.h"
#include "ctype.h"

#include "../common/book.h"

#define		N		1024
#define		LOGFILE		"gpu_vector.log"

FILE *fp;

__global__ void add(int *a,int *b,int *c) {
	int tid = threadIdx.x;
	printf("Thread ID: %i\n",tid);
	if (tid < N) {
		c[tid] = a[tid] + b[tid];
	}
	printf("ctid: %i\n",c[tid]);
}

void remove_log(){
	printf("Removing, older logfile\n");
	const char *cmd[] = { "find -type f -name", LOGFILE, "-delete" };
	char s[500] = {};
	sprintf(s, "%s '%s' %s", cmd[0], cmd[1], cmd[2]);
	printf("Executed: %s\n", s);
	system(s);
	free(cmd);
}


int main(void) {

	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;

	// allocate the memory on the GPU
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

	// fill the arrays ‘a’ and ‘b’ on the CPU
	for (int i=0; i<N; i++) {
		a[i] = i;
		b[i] = i * i;
	}

	// copy the arrays ‘a’ and ‘b’ to the GPU
	HANDLE_ERROR(cudaMemcpy(dev_a,a,N * sizeof(int),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b,b,N * sizeof(int),cudaMemcpyHostToDevice));
	add<<<1, N>>>(dev_a, dev_b, dev_c);

	// copy the array ‘c’ back from the GPU to the CPU
	HANDLE_ERROR(cudaMemcpy(c,dev_c,N * sizeof(int),cudaMemcpyDeviceToHost));

	fp = fopen(LOGFILE, "a");
	// display the results
	for (int i = 0; i < N; i++) {
		fprintf(fp, "%i %i %i\n", a[i], b[i], c[i]);
	}
	fclose(fp);
	// free the memory allocated on the GPU
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	return 0;
}
