#include <stdio.h>

__global__ void add(int a, int b, int *c) {
    *c = a + b;
}


int main( void ) {
    int c;
    int *dev_c;
    
    //Device Memory allocations
    cudaError_t err = cudaMalloc((void**)&dev_c, sizeof(&dev_c));
    if(err != cudaSuccess) {
	   printf("The error is %s\n", cudaGetErrorString(err));
    }

    add<<<1,1>>>(2, 7, dev_c);
    
    if(cudaPeekAtLastError() != cudaSuccess) {
	   printf("The error is %s\n", cudaGetErrorString(cudaGetLastError()));
    }


    cudaError_t err2 = cudaMemcpy( &c, dev_c, sizeof(c), cudaMemcpyDeviceToHost);
    if(err2 != cudaSuccess) {
	   printf("The error is %s\n", cudaGetErrorString(err2));
    }


    printf("2 + 7 = %d\n", c);
    cudaFree(dev_c);

    return 0;
}
