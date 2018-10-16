CC := g++
NVCC := nvcc
SRCDIR := src 
BUILDDIR := build
GPUSRCDIR := gpusrc
TARGET := bin

SRCEXT := cpp
GPUSRCEXT := cu
SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%$(SOURCES:.$(SRCEXT)=.o))
CFLAGS := -std=c++11
LINK := -L/usr/local/cuda/lib64
LIB := -lcudart
INC := -I include


$(BUILDDIR)/gpu.o: $(GPUSRCDIR)/%.$(GPUSRCEXT)


# all: main.o

# main.o:
# 	nvcc -c vector_add_gpu.cu -o vector_add_gpu.o
# 	g++ -std=c++11 -L/usr/local/cuda/lib64/ main.cpp vector_add_cpu.cpp vector_add_gpu.o -lcudart -o vector_add

clean:
	rm *.o
	rm vector_add