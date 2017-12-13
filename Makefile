PWD	:= $(shell pwd)

all: oil resize swirl

oil:
	mkdir out/oil -p
	make -C oil -j4

resize:
	mkdir out/resize -p
	make -C resize -j4

swirl:
	mkdir out/swirl -p
	make -C swirl -j4

clean:
	rm -rf out

.PHONY: all clean oil resize swirl

