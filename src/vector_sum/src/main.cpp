#include <stdlib.h>
#include <ctime>
#include <iostream>
#include <ratio>
#include <chrono>

#include "../include/vector_add_cpu.h"
#include "../include/vector_add_gpu.h"

using namespace std;
using namespace std::chrono;

/* Utility Functions */
// Allocate memory for array
void allocMemory(float** vec, int N) {
    *vec = new (nothrow) float[N];

    if (*vec == nullptr) {
        cout << "Error: Vector of length " << N;
        cout << " elements can't be allocated" << endl;
    }
}

// Fill vector with random values
void fillVector(float *vec, int N) {
    for (int i = 0; i < N; i++) {
        vec[i] = static_cast<float>(rand()) /
            (static_cast<float>(RAND_MAX/(50)));
    }
}

// Print vector values to stdout
void printVector(const float *vec, int N) {
    cout << "[ ";
    for (int i = 0; i < N; i++) {
    cout << vec[i] << " ";
    }
    cout << "]" << endl;
}

// void vectorAddCPU(float *a, float *b, float *c, int N) {
//  for(int i = 0; i < N; i++) {
//        c[i] = a[i] + b[i];
//   }
// }

double get_cpu_time() {
    return static_cast<double>(clock() / CLOCKS_PER_SEC);
}

int main(int argc, char** argv) {
    float *a;
    float *b;
    float *c;
    cout << "Enter desired vector length: ";

    int N;
    cin >> N;

    allocMemory(&a, N);
    allocMemory(&b, N);
    allocMemory(&c, N);

    srand(time(NULL));
    fillVector(a, N);
    fillVector(b, N);

    // cout << "Vector 1: ";
    // printVector(a, N);
    // cout << "Vector 2: ";
    // printVector(b, N);

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    vectorAddCPU(a, b, c, N);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

    // cout << "CPU Add Result: ";
    // printVector(c, N);
    cout << "CPU Time for " << N << " elements: " << time_span.count() * 1000
        << " msec" << endl;
    for (int i = 0; i < N; i++) {
        c[i] = 0;
    }
    vectorAddGPU_wrapper(a, b, c, N);

    delete[] a;
    delete[] b;
    delete[] c;
}
