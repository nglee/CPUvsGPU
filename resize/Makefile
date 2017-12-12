LDLIBS=-lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_gpu

all: main main_cuda

main: main.cpp
	g++ -std=c++11 $^ $(LDLIBS) -o $@

main_cuda: main.cpp
	g++ -std=c++11 $^ $(LDLIBS) -o $@ -DCUDA

clean:
	rm -f main main_cuda *resized*

.PHONY: all clean

