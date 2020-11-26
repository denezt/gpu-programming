#include "../common/book.h"
#include "stdlib.h"

#define		N	10
#define 	true	1 == 1


__global__ void add( int a, int b, int *c ) {
   *c = a + b;
   printf("GPU [add]: %i + %i = %d\n",a,b,*c);
}

int main( void ) {


	int a[255];
	int b[255];
	int c;
	int *dev_a;
	int *dev_b;
	int *dev_c;
	int counter = 0;	
	while (counter < 10) {
		HANDLE_ERROR( cudaMalloc( (void**)&dev_c, sizeof(int) ) );
		for (int i=0; i<N; i++) {
			a[i] = -i;
			b[i] = i * i;
			printf("a: %i\n", a[i]);
			printf("b: %i\n", b[i]);
			add<<<N,1>>>( a[i], b[i], dev_c );			
		}


		HANDLE_ERROR(cudaMemcpy(&c,dev_c,sizeof(int),cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaFree(dev_c));

		// free the memory allocated on the GPU    
		cudaFree(&dev_a);
		cudaFree(&dev_b);
		cudaFree(&dev_c);
		counter++;
	}
	return 0;
}


/** END OF CODE **/



