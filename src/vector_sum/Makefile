all: main.o

main.o:
	nvcc -c vector_add_gpu.cu -o vector_add_gpu.o
	g++ -std=c++11 -L/usr/local/cuda/lib64/ main.cpp vector_add_cpu.cpp vector_add_gpu.o -lcudart -o vector_add

clean:
	rm *.o
	rm vector_add