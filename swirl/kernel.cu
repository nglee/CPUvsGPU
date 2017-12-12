/*
 * Algorithm copied from "supercomputingblog.com".
 * (http://supercomputingblog.com/openmp/image-twist-and-swirl-algorithm/2/)
 */

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <stdio.h>

#define C_PI 3.141592653589793238462643383279502884197169399375f

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "%s\nFile: %s\nLine Number: %d\nReason: %s\n",
                msg, file_name, line_number, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

#define SAFE_CUDA_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

__global__ void
swirl_cuda(unsigned char* img, const unsigned char* img_copy, const int step, const int width,
           const int height, const float cX, const float cY, const float factor)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= height || j >= width)
        return;

    float relY = cY - i;
    float relX = j - cX;

    float originalAngle;

    if (relX != 0) {
        originalAngle = atanf(abs(relY)/abs(relX));
        if (relX > 0 && relY < 0) originalAngle = 2.0f * C_PI - originalAngle;
        else if (relX < 0 && relY >= 0) originalAngle = C_PI - originalAngle;
        else if (relX < 0 && relY < 0) originalAngle += C_PI;
    } else {
        if (relY >= 0) originalAngle = 0.5f * C_PI;
        else originalAngle = 1.5f * C_PI;
    }

    float radius = sqrtf(relX * relX + relY * relY);
    float radius_full = sqrtf(cX * cX + cY * cY);

    //float newAngle = originalAngle + 1/(factor * radius + (4.0f / C_PI));
    float newAngle = originalAngle + factor * radius / radius_full;

    int srcX = radius * cos(newAngle) + 0.5f;
    int srcY = radius * sin(newAngle) + 0.5f;

    srcX += cX;
    srcY += cY;
    srcY = height - srcY;

    if (srcX < 0) srcX = 0;
    else if (srcX >= width) srcX = width - 1;
    if (srcY < 0) srcY = 0;
    else if (srcY >= height) srcY = height - 1;

    img[i * step + 3 * j] = img_copy[srcY * step + 3 * srcX];
    img[i * step + 3 * j + 1] = img_copy[srcY * step + 3 * srcX + 1];
    img[i * step + 3 * j + 2] = img_copy[srcY * step + 3 * srcX + 2];
}

void swirl_cuda_wrapper(cv::Mat& img, float factor)
{
    int width = img.cols;
    int height = img.rows;

    float cX = (width - 1) / 2.0f;
    float cY = (height - 1) / 2.0f;

    unsigned char* d_img;
    unsigned char* d_img_copy;

    const int img_size = img.step * img.rows;

    SAFE_CUDA_CALL(cudaMalloc<unsigned char>(&d_img, img_size), "CUDA Malloc Failed");
    SAFE_CUDA_CALL(cudaMalloc<unsigned char>(&d_img_copy, img_size), "CUDA Malloc Failed");

    SAFE_CUDA_CALL(cudaMemcpy(d_img, img.ptr(), img_size, cudaMemcpyHostToDevice),
            "CUDA Memcpy Host To Device Failed");
    SAFE_CUDA_CALL(cudaMemcpy(d_img_copy, d_img, img_size, cudaMemcpyDeviceToDevice),
            "CUDA Memcpy Device To Device Failed");

    const dim3 threadsPerBlock(32, 32); /* upper limit : 1024 */
    const dim3 blocksPerGrid(
            (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    swirl_cuda<<<blocksPerGrid, threadsPerBlock>>>(d_img, d_img_copy, img.step,
            width, height, cX, cY, factor);

    SAFE_CUDA_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");
    SAFE_CUDA_CALL(cudaMemcpy(img.ptr(), d_img, img_size, cudaMemcpyDeviceToHost),
            "CUDA Memcpy Device To Host Failed");

    SAFE_CUDA_CALL(cudaFree(d_img_copy), "CUDA Free Failed");
    SAFE_CUDA_CALL(cudaFree(d_img), "CUDA Free Failed");
}
