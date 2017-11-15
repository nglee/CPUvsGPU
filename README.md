### Purpose
To study execution speed difference between CPU and GPU.

### Notes
Tested on NVIDIA Jetson TX1 Development Kit.

### Descriptions

###### ./images
* Test images.
* Images are from [pixabay](https://pixabay.com/).

###### ./oil
* Makes an image look like an oil painting.
* Original CPU algorithm is from [The Supercomputing Blog](http://supercomputingblog.com/graphics/oil-painting-algorithm/).

###### ./swirl
* Twists an image.
* Original CPU algorithm is from [The Supercomputing Blog](http://supercomputingblog.com/openmp/image-twist-and-swirl-algorithm/)

###### ./resize
* Runs CPU and GPU version of OpenCV `resize()` function.
* Used OpenCV4Tegra.
* For the GPU version, only the computation speed is measured, so memory copying delays beween host and device memory are neglected.




