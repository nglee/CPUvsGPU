#
#  Author : nglee
#  E-mail : lee.namgoo@sualab.com
#

INC_DIRS=

LIB_DIRS=

LIBS=-lopencv_core -lopencv_highgui

CPP_OPTIONS=-std=c++11 -O2

NVCC_OPTIONS=-arch=sm_53 -ccbin=g++ -std=c++11 -w -O2

TARGETS=main main_omp main_cuda main_cuda_shared main_cuda_second

.PHONY: all clean

all: $(TARGETS)

main: main.cpp
	g++ $(CPP_OPTIONS) $(INC_DIRS) $(LIB_DIRS) $< $(LIBS) -o $@

main_omp: main.cpp
	g++ $(CPP_OPTIONS) -fopenmp -DOMP $(INC_DIRS) $(LIB_DIRS) $< $(LIBS) -o $@

main_cuda: main.cpp kernel.cu
	nvcc -DCUDA $(NVCC_OPTIONS) $(INC_DIRS) $(LIB_DIRS) $^ $(LIBS) -o $@

main_cuda_shared: main.cpp kernel.cu
	nvcc -DCUDA -DCUDA_SHARED $(NVCC_OPTIONS) $(INC_DIRS) $(LIB_DIRS) $^ $(LIBS) -o $@

main_cuda_second: main.cpp kernel.cu
	nvcc -DCUDA -DCUDA_SECOND_VERSION $(NVCC_OPTIONS) $(INC_DIRS) $(LIB_DIRS) $^ $(LIBS) -o $@

clean:
	rm $(TARGETS) *oil*
