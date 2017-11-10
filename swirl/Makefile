#
#  Author : nglee
#  E-mail : lee.namgoo@sualab.com
#

INC_DIRS=

LIB_DIRS=

LIBS=-lopencv_core -lopencv_highgui

NVCC_OPTIONS=-arch=sm_53 -ccbin=g++ -std=c++11 -w

.PHONY: all clean

all: main main_omp main_cuda

main: main.cpp
	g++ -std=c++11 $(INC_DIRS) $(LIB_DIRS) $< $(LIBS) -o $@

main_omp: main.cpp
	g++ -std=c++11 -fopenmp $(INC_DIRS) $(LIB_DIRS) $< $(LIBS) -o $@

main_cuda: main.cpp kernel.cu
	nvcc -DCUDA $(NVCC_OPTIONS) $(INC_DIRS) $(LIB_DIRS) $^ $(LIBS) -o $@

clean:
	rm main main_omp main_cuda
