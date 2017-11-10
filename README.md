## Purpose

To study execution speed difference between CPU and GPU.

## Notes

Tested on NVIDIA Jetson TX1 Development Kit.


## Descriptions

./images

Test data.

Images are from pixabay(https://pixabay.com/).

./oil

Algorithm from supercomputingblog.com

(http://supercomputingblog.com/graphics/oil-painting-algorithm/)

CPU vs. CPU with OpenMP vs. CUDA w/o shared memory vs. CUDA w/ sharedmemory vs. CUDA second version

./resize

Runs CPU and GPU version of OpenCV `resize()` function.

Tested with OpenCV4Tegra.

Only the computation speed is measured, so memory copying delays beween host and device memory are neglected.




