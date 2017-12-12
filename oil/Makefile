LDLIBS=-lopencv_core -lopencv_highgui
CXXFLAGS=-std=c++11 -O2
NVCCFLAGS=-arch=sm_53 -ccbin=g++ -std=c++11 -O2
TARGETS=main main_omp main_cuda main_cuda_shared main_cuda_second

all: $(TARGETS)

main: main.cpp
	g++ $(CXXFLAGS) $^ $(LDLIBS) -o $@

main_omp: main.cpp
	g++ $(CXXFLAGS) -fopenmp -DOMP $^ $(LDLIBS) -o $@

main_cuda: main.cpp kernel.cu
	nvcc -DCUDA $(NVCCFLAGS) $^ $(LDLIBS) -o $@

main_cuda_shared: main.cpp kernel.cu
	nvcc -DCUDA -DCUDA_SHARED $(NVCCFLAGS) $^ $(LDLIBS) -o $@

main_cuda_second: main.cpp kernel.cu
	nvcc -DCUDA -DCUDA_SECOND_VERSION $(NVCCFLAGS) $^ $(LDLIBS) -o $@

clean:
	rm -f $(TARGETS) *oil*

.PHONY: all clean

