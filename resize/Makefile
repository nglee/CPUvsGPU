#
#  Author : nglee
#  E-mail : lee.namgoo@sualab.com
#

INCDIRS=

LIBDIRS=

LIBS=-lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_gpu

.PHONY: all clean

all: main main_cuda

main: main.cpp
	g++ -std=c++11 $(INCDIRS) $(LIBDIRS) $< $(LIBS) -o $@

main_cuda: main.cpp
	g++ -std=c++11 $(INCDIRS) $(LIBDIRS) $< $(LIBS) -o $@ -DCUDA

clean:
	rm main main_cuda *resized*
