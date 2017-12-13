CXX			= g++
NVCC		= nvcc

CXXFLAGS	= -std=c++11 -O2
CUFLAGS		= -arch=sm_53 -ccbin=g++ -std=c++11 -O2

LDLIBS		= -lopencv_core -lopencv_highgui

OUTDIR		= ../out/oil
OBJECTS		= oil_CPU oil_CPU_OMP oil_CUDA oil_CUDA_SHARED oil_CUDA_SECOND test.sh
TARGETS 	= $(OBJECTS:%=$(OUTDIR)/%)

all: $(TARGETS)

$(OUTDIR)/oil_CPU: main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(OUTDIR)/oil_CPU_OMP: main.cpp
	$(CXX) -DOMP $(CXXFLAGS) -fopenmp -o $@ $^ $(LDLIBS)

$(OUTDIR)/oil_CUDA: main.cpp kernel.cu
	nvcc $(CUFLAGS) -DCUDA -o $@ $^ $(LDLIBS)

$(OUTDIR)/oil_CUDA_SHARED: main.cpp kernel.cu
	nvcc $(CUFLAGS) -DCUDA -DCUDA_SHARED -o $@ $^ $(LDLIBS)

$(OUTDIR)/oil_CUDA_SECOND: main.cpp kernel.cu
	nvcc $(CUFLAGS) -DCUDA -DCUDA_SECOND_VERSION -o $@ $^ $(LDLIBS)

$(OUTDIR)/test.sh: test.sh
	cp test.sh $(OUTDIR)

.PHONY: all

