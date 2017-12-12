LDLIBS=-lopencv_core -lopencv_highgui
CXXFLAGS=-std=c++11 -O2
NVCCFLAGS=-arch=sm_53 -ccbin=g++ -std=c++11 -O2
TARGETS=main main_omp main_cuda

all: $(TARGETS)

main: main.cpp
	g++ $(CXXFLAGS) $^ $(LDLIBS) -o $@

main_omp: main.cpp
	g++ $(CXXFLAGS) -fopenmp -DOMP $^ $(LDLIBS) -o $@

main_cuda: main.cpp kernel.cu
	nvcc -DCUDA $(NVCCFLAGS) $^ $(LDLIBS) -o $@

clean:
	rm -f $(TARGETS) *twisted*

.PHONY: all clean

