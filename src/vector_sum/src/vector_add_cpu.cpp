#include "../include/vector_add_cpu.h"

void vectorAddCPU(float *a, float *b, float *c, int N) {
	for(int i = 0; i < N; i++) {
		c[i] = a[i] + b[i];
	}
}