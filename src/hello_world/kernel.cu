#include <iostream>
#include <stdio.h>

// __global__ Compiler runs this code on device not host
__global__ void kernel( void ) {
}

int main( void ) {
    kernel<<<1,1>>>();
    printf( "Hello, World!\n" );
    return 0;
}

