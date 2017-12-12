all:
	make -C oil -j4
	make -C resize -j4
	make -C swirl -j4

clean:
	make -C oil clean
	make -C resize clean
	make -C swirl clean

.PHONY: all clean
