CXX			= g++

CXXFLAGS	= -std=c++11 -O2

LDLIBS		=-lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_gpu

OUTDIR		= ../out/resize
OBJECTS		= resize_CPU resize_CUDA test.sh
TARGETS 	= $(OBJECTS:%=$(OUTDIR)/%)

all: $(TARGETS)

$(OUTDIR)/resize_CPU: main.cpp
	g++ $(CXXFLAGS) $^ $(LDLIBS) -o $@

$(OUTDIR)/resize_CUDA: main.cpp
	g++ -DCUDA $(CXXFLAGS) $^ $(LDLIBS) -o $@ -DCUDA

$(OUTDIR)/test.sh: test.sh
	cp test.sh $(OUTDIR)

.PHONY: all

