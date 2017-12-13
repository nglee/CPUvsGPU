CXX			= g++
NVCC		= nvcc

CXXFLAGS	= -std=c++11 -O2
CUFLAGS		= -arch=sm_53 -ccbin=g++ -std=c++11 -O2

LDLIBS		= -lopencv_core -lopencv_highgui

OUTDIR		= ../out/swirl
OBJECTS		= swirl_CPU swirl_CPU_OMP swirl_CUDA test.sh
TARGETS 	= $(OBJECTS:%=$(OUTDIR)/%)

all: $(TARGETS)

$(OUTDIR)/swirl_CPU: main.cpp
	$(CXX) $(CXXFLAGS) $^ $(LDLIBS) -o $@

$(OUTDIR)/swirl_CPU_OMP: main.cpp
	$(CXX) -DOMP $(CXXFLAGS) -fopenmp $^ $(LDLIBS) -o $@

$(OUTDIR)/swirl_CUDA: main.cpp kernel.cu
	$(NVCC) -DCUDA $(CUFLAGS) $^ $(LDLIBS) -o $@

$(OUTDIR)/test.sh: test.sh
	cp test.sh $(OUTDIR)

.PHONY: all

