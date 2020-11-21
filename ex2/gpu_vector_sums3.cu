#include "../common/book.h"
// #include "../common/book.h"

#define		N	10
#define 	true	1 == 1


__global__ void add( int a, int b, int *c ) {
   *c = a + b;
   printf("GPU [add]: %i + %i = %d\n",a,b,*c);
}

__global__ void subtract( int a, int b, int *c ) {
   *c = a - b;
   printf("GPU [subtract]: %i - %i = %d\n",a,b,*c);
}

int main( void ) {
	int c;
	int *dev_c;
	int counter = 0;
	while (counter < 10) {
		HANDLE_ERROR( cudaMalloc( (void**)&dev_c, sizeof(int) ) );

		add<<<N,1>>>( 2, 7, dev_c );
			HANDLE_ERROR( cudaMemcpy( &c, dev_c, sizeof(int),
		      cudaMemcpyDeviceToHost ) );
		// printf( "2 + 7 = %d\n", c );
		HANDLE_ERROR( cudaFree( dev_c ) );

		// free the memory allocated on the GPU    
		// cudaFree( &dev_a );
		// cudaFree( &dev_b );
		cudaFree( &dev_c );
		counter++;
	}
	
	while (counter > 0) {
		HANDLE_ERROR( cudaMalloc( (void**)&dev_c, sizeof(int) ) );

		subtract<<<N,1>>>( 2, 7, dev_c );
			HANDLE_ERROR( cudaMemcpy( &c, dev_c, sizeof(int),
		      cudaMemcpyDeviceToHost ) );
		// printf( "2 + 7 = %d\n", c );
		HANDLE_ERROR( cudaFree( dev_c ) );

		// free the memory allocated on the GPU    
		// cudaFree( &dev_a );
		// cudaFree( &dev_b );
		cudaFree( &dev_c );
		counter--;
	}

	return 0; 
}


/** END OF CODE **/



