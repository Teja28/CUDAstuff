CC := g++
NVCC := nvcc
SRCDIR := src 
BUILDDIR := build
TARGET := bin/app

SRCEXT := cpp
GPUSRCEXT := cu
SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
GPUSOURCES := $(shell find $(SRCDIR) -type f -name *.$(GPUSRCEXT))
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
GPUOBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(GPUSRCEXT)=.o))
CFLAGS := -std=c++11
LINK := -L/usr/local/cuda/lib64
LIB := -lcudart
INC := -I include

$(TARGET): build/vector_add_gpu.o $(SOURCES)
	$(CC) $(CFLAGS) $(LINK) $^ $(LIB) -o $(TARGET)

# Had trouble with this rule, only way I could get compilation
$(BUILDDIR)/vector_add_gpu.o: src/vector_add_gpu.cu
	@mkdir -p $(BUILDDIR)
	$(NVCC) -c -o $@ $<

clean:
	$(RM) -r $(BUILDDIR) $(TARGET)

# all: main.o

# main.o:
# 	nvcc -c vector_add_gpu.cu -o vector_add_gpu.o
# 	g++ -std=c++11 -L/usr/local/cuda/lib64/ main.cpp vector_add_cpu.cpp vector_add_gpu.o -lcudart -o vector_add

# clean:
# 	rm *.o
# 	rm vector_add