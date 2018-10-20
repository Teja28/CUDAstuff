#include <iostream>
#include "../include/vector_add_gpu.h"
using namespace std;


void checkCUDAError(cudaError_t err) {
	if(err != cudaSuccess) {
		cout << "The error is " << cudaGetErrorString(cudaGetLastError()) << ".";
		cout << endl;
	}
}

__global__ void vectorAddGPU(float *a, float *b, float *c, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < N) {
		c[i] = a[i] + b[i];
	}
}

void vectorAddGPU_wrapper(float *h_a, float *h_b, float h_c[], int N) {
	float *dev_a;
	float *dev_b;
	float *dev_c;
	size_t size = N * sizeof(float);

	float ms0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Allocate GPU memory
	checkCUDAError(cudaMalloc(&dev_a, size));
	checkCUDAError(cudaMalloc(&dev_b, size));
	checkCUDAError(cudaMalloc(&dev_c, size));

	// Copy host data to device
	checkCUDAError(cudaMemcpy(dev_a, h_a, size, cudaMemcpyHostToDevice));
	checkCUDAError(cudaMemcpy(dev_b, h_b, size, cudaMemcpyHostToDevice));
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms0, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cout << "GPU Memory Allocation/Copy to Device: " << ms0 << " ms" << endl;
	float ms = ms0;

	int blockSize;
	int gridSize;
	if(N < 1024) {
		blockSize = N;
		gridSize = 1;
	}
	else {
		blockSize = 1024;
		gridSize = (int)ceil((float)N / blockSize); 
	}

	cudaEvent_t start1, stop1;
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, 0);
	vectorAddGPU<<<gridSize, blockSize>>>(dev_a, dev_b, dev_c, N);
	cudaEventRecord(stop1, 0);
	cudaEventSynchronize(stop1);
	float ms1;
	cudaEventElapsedTime(&ms1, start1, stop1);
	cudaEventDestroy(start1);
	cudaEventDestroy(stop1);
	cout << "Computation Time: " << ms1 << " ms" << endl;
	ms += ms1;

	cudaEvent_t start2, stop2;
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);
	cudaEventRecord(start2, 0);
	checkCUDAError(cudaMemcpy(h_c, dev_c, size, cudaMemcpyDeviceToHost));
	checkCUDAError(cudaFree(dev_a));
	checkCUDAError(cudaFree(dev_b));
	checkCUDAError(cudaFree(dev_c));
	cudaEventRecord(stop2, 0);
	cudaEventSynchronize(stop2);
	float ms2;
	cudaEventElapsedTime(&ms2, start, stop);
	cudaEventDestroy(start2);
	cudaEventDestroy(stop2);
	cout << "GPU Memory Deallocation/Copy to Host: " << ms2 << " ms" << endl;
	cout << "Total GPU Time: " << ms << " ms" << endl;
}

// int main() {
// 	cout << "Enter desired vector length: ";
// 	int N;
//     cin >> N;

//     float *a;
//     float *b;
//     float *c;

//     allocMemory(&a, N);
//     allocMemory(&b, N);
//     allocMemory(&c, N);

//     srand(time(NULL));
//     fillVector(a, N);
//     fillVector(b, N);

//     vectorAddGPU_wrapper(a, b, c, N);
	
// 	printVector(c, N);
// }