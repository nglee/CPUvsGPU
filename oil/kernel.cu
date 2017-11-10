/*
 * Algorithm is from "supercomputingblog.com".
 * (http://supercomputingblog.com/graphics/oil-painting-algorithm/)
 *
 * Author : nglee
 * E-mail : lee.namgoo@sualab.com
 */

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <stdio.h>

#ifdef CUDA_SECOND_VERSION
#  define BLOCKDIMX_KERNEL1   32
#  define BLOCKDIMY_KERNEL1   32
#  define THREAD_COVERAGE_X   4   /* # pixels in a row that 1 thread covers */
#  define THREAD_COVERAGE_Y   1   /* # rows of pixels that 1 thread covers */
#  define BLOCKDIMX_KERNEL2   8
#  define BLOCKDIMY_KERNEL2   4
#else
#  define BLOCKDIMX   8
#  define BLOCKDIMY   4
#endif

static inline void _safe_cuda_call(cudaError err,
        const char* file_name, const int line_number)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "File: %s\nLine Number: %d\nReason: %s\n",
                file_name, line_number, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

#define SAFE_CUDA_CALL(call) _safe_cuda_call((call), __FILE__, __LINE__)
#define CROP(X) X > 255U ? 255U : X;

__constant__ int d_width;
__constant__ int d_height;
__constant__ int d_step;
__constant__ int d_radius;
__constant__ int d_intensity_level;

#ifdef CUDA_SECOND_VERSION
__global__ void
oil_cuda_kernel_1(unsigned char* img, unsigned int* intensity)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= d_height || j >= d_width)
        return;

    unsigned int r = (unsigned int)img[i * d_step + j * 3];
    unsigned int g = (unsigned int)img[i * d_step + j * 3 + 1];
    unsigned int b = (unsigned int)img[i * d_step + j * 3 + 2];

    intensity[i * d_step + j] = (r+g+b)*d_intensity_level/766;
}
__global__ void
oil_cuda_kernel_2(unsigned char* img, unsigned char* img_copy, unsigned int* intensity)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    int start_row = i * THREAD_COVERAGE_Y;
    int start_col = j * THREAD_COVERAGE_X;

    if (start_row >= d_height || start_col >= d_width)
        return;

#if THREAD_COVERAGE_Y != 1
#  error "Current implementation only supports THREAD_COVERAGE_Y == 1"
#endif

    unsigned int buf[80] = { 0 };
    unsigned int *intensity_count   = buf;
    unsigned int *average_r         = buf + d_intensity_level;
    unsigned int *average_g         = buf + d_intensity_level * 2;
    unsigned int *average_b         = buf + d_intensity_level * 3;

    {
        // histogram for the starting pixel
        for (int k = start_row - d_radius; k <= start_row + d_radius; k++) {
            if (k < 0) continue;
            if (k >= d_height) break;

            for (int l = start_col - d_radius; l <= start_col + d_radius; l++) {
                if (l < 0) continue;
                if (l >= d_width) break;

                unsigned int cur_intensity = intensity[k * d_step + l];
                intensity_count[cur_intensity]++;
                average_r[cur_intensity] += (unsigned int)img_copy[k * d_step + l * 3];
                average_g[cur_intensity] += (unsigned int)img_copy[k * d_step + l * 3 + 1];
                average_b[cur_intensity] += (unsigned int)img_copy[k * d_step + l * 3 + 2];
            }
        }

        // value for the starting pixel
        unsigned int max = 0;
        unsigned int max_index = 0;

        for (int k = 0; k < d_intensity_level; k++)
            if (intensity_count[k] > max) {
                max = intensity_count[k];
                max_index = k;
            }

        if (max == 0) {
            printf("i = %d, j = %d, (%d, %d) max = %u, max_index = %u\n",
                    i, j, start_row, start_col, max, max_index);

            for (int k = start_row - d_radius; k <= start_row + d_radius; k++) {
                if (k < 0) continue;
                if (k >= d_height) break;
                for (int l = start_col - d_radius; l <= start_col + d_radius; l++) {
                    if (l < 0) continue;
                    if (l >= d_width) break;
                    printf("intensity(%d, %d) = %u\n", k, l, intensity[k * d_step + l]);
                }
            }
        }

        unsigned int final_r = average_r[max_index] / max;
        unsigned int final_g = average_g[max_index] / max;
        unsigned int final_b = average_b[max_index] / max;

        img[start_row * d_step + start_col * 3] = CROP(final_r);
        img[start_row * d_step + start_col * 3 + 1] = CROP(final_g);
        img[start_row * d_step + start_col * 3 + 2] = CROP(final_b);
    }

    // update histogram and calculate values for the rest pixels
    for (int l = start_col + 1; l < start_col + THREAD_COVERAGE_X; l++) {
        if (l >= d_width)
            break;

        int remove_col = l - 1 - d_radius;
        int add_col = l + d_radius;

        if (remove_col >= 0) {
            for (int k = start_row - d_radius; k <= start_row + d_radius; k++) {
                if (k < 0) continue;
                if (k >= d_height) break;

                unsigned int cur_intensity = intensity[k * d_step + remove_col];
                intensity_count[cur_intensity]--;
                average_r[cur_intensity] -= (unsigned int)img_copy[k * d_step + remove_col * 3];
                average_g[cur_intensity] -= (unsigned int)img_copy[k * d_step + remove_col * 3 + 1];
                average_b[cur_intensity] -= (unsigned int)img_copy[k * d_step + remove_col * 3 + 2];
            }
        }

        if (add_col < d_width) {
            for (int k = start_row - d_radius; k <= start_row + d_radius; k++) {
                if (k < 0) continue;
                if (k >= d_height) break;

                unsigned int cur_intensity = intensity[k * d_step + add_col];
                intensity_count[cur_intensity]++;
                average_r[cur_intensity] += (unsigned int)img_copy[k * d_step + add_col * 3];
                average_g[cur_intensity] += (unsigned int)img_copy[k * d_step + add_col * 3 + 1];
                average_b[cur_intensity] += (unsigned int)img_copy[k * d_step + add_col * 3 + 2];
            }
        }

        unsigned int max = 0;
        unsigned int max_index = 0;

        for (int k = 0; k < d_intensity_level; k++)
            if (intensity_count[k] > max) {
                max = intensity_count[k];
                max_index = k;
            }

        unsigned int final_r = average_r[max_index] / max;
        unsigned int final_g = average_g[max_index] / max;
        unsigned int final_b = average_b[max_index] / max;

        img[start_row * d_step + l * 3] = CROP(final_r);
        img[start_row * d_step + l * 3 + 1] = CROP(final_g);
        img[start_row * d_step + l * 3 + 2] = CROP(final_b);
    }
}

#else   // !CUDA_SECOND_VERSION
#  ifndef CUDA_SHARED
__global__ void
oil_cuda(unsigned char* img, unsigned char* img_copy)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= d_height || j >= d_width)
        return;

    unsigned int buf[80] = { 0 };
    unsigned int *intensity_count   = buf;
    unsigned int *average_r         = buf + d_intensity_level;
    unsigned int *average_g         = buf + d_intensity_level * 2;
    unsigned int *average_b         = buf + d_intensity_level * 3;

    //memset(buf, 0, sizeof(unsigned int) * d_intensity_level * 4);

    for (int k = i-d_radius; k <= i+d_radius; k++) {
        if (k<0 || k>=d_height)
            continue;
        for (int l = j-d_radius; l <= j+d_radius; l++) {
            if (l<0 || l>=d_width)
                continue;

            unsigned int r = (unsigned int)img_copy[k * d_step + l * 3];
            unsigned int g = (unsigned int)img_copy[k * d_step + l * 3 + 1];
            unsigned int b = (unsigned int)img_copy[k * d_step + l * 3 + 2];

            unsigned int cur_intensity = (r+g+b)*d_intensity_level/766;
            intensity_count[cur_intensity]++;
            average_r[cur_intensity] += r;
            average_g[cur_intensity] += g;
            average_b[cur_intensity] += b;
        }
    }

    unsigned int max = 0;
    unsigned int max_index = 0;

    for (int k = 0; k < d_intensity_level; k++)
        if (intensity_count[k] > max) {
            max = intensity_count[k];
            max_index = k;
        }

    unsigned int final_r = average_r[max_index] / max;
    unsigned int final_g = average_g[max_index] / max;
    unsigned int final_b = average_b[max_index] / max;

    img[i * d_step + j * 3] = CROP(final_r);
    img[i * d_step + j * 3 + 1] = CROP(final_g);
    img[i * d_step + j * 3 + 2] = CROP(final_b);
}
#  else // CUDA_SHARED
extern __shared__ unsigned int s_intensity[]; // dynamic shared memory

__device__ inline void
calc_intensity(const int y, const int x, const unsigned char* img_copy)
{
    unsigned int r = img_copy[y * d_step + x * 3];
    unsigned int g = img_copy[y * d_step + x * 3 + 1];
    unsigned int b = img_copy[y * d_step + x * 3 + 2];

    unsigned int intensity = (r+g+b)*d_intensity_level/766;

    int i = (y - blockIdx.y * blockDim.y) + d_radius;
    int j = (x - blockIdx.x * blockDim.x) + d_radius;
    s_intensity[i * (blockDim.x + 2*d_radius) + j] = intensity;
}

__global__ void
oil_cuda(unsigned char* img, unsigned char* img_copy)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= d_height || j >= d_width)
        return;

    if (threadIdx.y == 0) {
        if (threadIdx.x == 0) {
            // left-top
            for (int k = i-d_radius; k <= i; k++) {
                if (k<0) continue;
                for (int l = j-d_radius; l <= j; l++) {
                    if (l<0) continue;
                    calc_intensity(k, l, img_copy);
                }
            }
        } else if (threadIdx.x == blockDim.x - 1) {
            // right-top
            for (int k = i-d_radius; k <= i; k++) {
                if (k<0) continue;
                for (int l = j; l <= j+d_radius; l++) {
                    if (l>=d_width) continue;
                    calc_intensity(k, l, img_copy);
                }
            }
        } else {
            // top
            for (int k = i-d_radius; k <= i; k++) {
                if (k<0) continue;
                calc_intensity(k, j, img_copy);
            }
        }
    } else if (threadIdx.y == blockDim.y - 1) {
        if (threadIdx.x == 0) {
            // left-bottom
            for (int k = i; k <= i+d_radius; k++) {
                if (k>=d_height) continue;
                for (int l = j-d_radius; l <= j; l++) {
                    if (l<0) continue;
                    calc_intensity(k, l, img_copy);
                }
            }
        } else if (threadIdx.x == blockDim.x - 1) {
            // right-bottom
            for (int k = i; k <= i+d_radius; k++) {
                if (k>=d_height) continue;
                for (int l = j; l <= j+d_radius; l++) {
                    if (l>=d_width) continue;
                    calc_intensity(k, l, img_copy);
                }
            }
        } else {
            // bottom
            for (int k = i; k <= i+d_radius; k++) {
                if (k>=d_height) continue;
                calc_intensity(k, j, img_copy);
            }
        }
    } else {
        if (threadIdx.x == 0) {
            // left
            for (int l = j-d_radius; l <= j; l++) {
                if (l<0) continue;
                calc_intensity(i, l, img_copy);
            }
        } else if (threadIdx.x == blockDim.x - 1) {
            // right
            for (int l = j; l <= j+d_radius; l++) {
                if (l>=d_width) continue;
                calc_intensity(i, l, img_copy);
            }
        } else {
            // center
            calc_intensity(i, j, img_copy);
        }
    }

    unsigned int buf[80] = { 0 };
    unsigned int *intensity_count   = buf;
    unsigned int *average_r         = buf + d_intensity_level;
    unsigned int *average_g         = buf + d_intensity_level * 2;
    unsigned int *average_b         = buf + d_intensity_level * 3;

    //memset(buf, 0, sizeof(unsigned int) * d_intensity_level * 4);

    __syncthreads(); /* sync among threads in a block */

    for (int k = i-d_radius; k <= i+d_radius; k++) {
        if (k<0 || k>=d_height)
            continue;
        for (int l = j-d_radius; l <= j+d_radius; l++) {
            if (l<0 || l>=d_width)
                continue;

            int y = (k - blockIdx.y * blockDim.y) + d_radius;
            int x = (l - blockIdx.x * blockDim.x) + d_radius;

            unsigned int cur_intensity = s_intensity[y * (blockDim.x + 2*d_radius) + x];
            intensity_count[cur_intensity]++;
            average_r[cur_intensity] += (unsigned int)img_copy[k * d_step + l * 3];
            average_g[cur_intensity] += (unsigned int)img_copy[k * d_step + l * 3 + 1];
            average_b[cur_intensity] += (unsigned int)img_copy[k * d_step + l * 3 + 2];
        }
    }

    unsigned int max = 0;
    unsigned int max_index = 0;

    for (int k = 0; k < d_intensity_level; k++)
        if (intensity_count[k] > max) {
            max = intensity_count[k];
            max_index = k;
        }

    unsigned int final_r = average_r[max_index] / max;
    unsigned int final_g = average_g[max_index] / max;
    unsigned int final_b = average_b[max_index] / max;

    img[i * d_step + j * 3] = CROP(final_r);
    img[i * d_step + j * 3 + 1] = CROP(final_g);
    img[i * d_step + j * 3 + 2] = CROP(final_b);
}
#  endif // CUDA_SHARED
#endif  // !CUDA_SECOND_VERSION

void oil_cuda_wrapper(cv::Mat& h_img, const int radius, const int intensity_level)
{
    int width = h_img.cols;
    int height = h_img.rows;
    int step = h_img.step;

    unsigned char* d_img;
    unsigned char* d_img_copy;

    const int h_img_size = step * height;

    SAFE_CUDA_CALL(cudaMalloc<unsigned char>(&d_img, h_img_size));
    SAFE_CUDA_CALL(cudaMalloc<unsigned char>(&d_img_copy, h_img_size));

    SAFE_CUDA_CALL(cudaMemcpy(d_img, h_img.ptr(), h_img_size, cudaMemcpyHostToDevice));
    SAFE_CUDA_CALL(cudaMemcpy(d_img_copy, d_img, h_img_size, cudaMemcpyDeviceToDevice));

    SAFE_CUDA_CALL(cudaMemcpyToSymbol(d_width, &width, sizeof(int), 0, cudaMemcpyHostToDevice));
    SAFE_CUDA_CALL(cudaMemcpyToSymbol(d_height, &height, sizeof(int), 0, cudaMemcpyHostToDevice));
    SAFE_CUDA_CALL(cudaMemcpyToSymbol(d_step, &step, sizeof(int), 0, cudaMemcpyHostToDevice));
    SAFE_CUDA_CALL(cudaMemcpyToSymbol(d_radius, &radius, sizeof(int), 0, cudaMemcpyHostToDevice));
    SAFE_CUDA_CALL(cudaMemcpyToSymbol(d_intensity_level, &intensity_level, sizeof(int), 0,
                cudaMemcpyHostToDevice));

#ifdef CUDA_SECOND_VERSION

    unsigned int* d_intensity;
    SAFE_CUDA_CALL(cudaMalloc<unsigned int>(&d_intensity, step * height * sizeof(unsigned int)));

    // first kernel calculates each pixel's intensity value
    {
        const dim3 threadsPerBlock(BLOCKDIMX_KERNEL1, BLOCKDIMY_KERNEL1);
        const dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                                 (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
        oil_cuda_kernel_1<<<blocksPerGrid, threadsPerBlock>>>(d_img, d_intensity);
    }

    // second kernel builds histograms for each pixel and calculate the value for each pixel
    {
        const dim3 threadsPerBlock(BLOCKDIMX_KERNEL2, BLOCKDIMY_KERNEL2);
        const dim3 blocksPerGrid(((width + THREAD_COVERAGE_X - 1) / THREAD_COVERAGE_X
                                  + threadsPerBlock.x - 1) / threadsPerBlock.x,
                                 ((height + THREAD_COVERAGE_Y - 1) / THREAD_COVERAGE_Y
                                  + threadsPerBlock.y - 1) / threadsPerBlock.y);
        oil_cuda_kernel_2<<<blocksPerGrid, threadsPerBlock>>>(d_img, d_img_copy, d_intensity);
    }

#else   // !CUDA_SECOND_VERSION

    const dim3 threadsPerBlock(BLOCKDIMX, BLOCKDIMY);
    const dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                             (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

#  ifdef CUDA_SHARED

    const size_t smem_size = sizeof(unsigned int)*(BLOCKDIMX + 2*radius)*(BLOCKDIMY + 2*radius);
    oil_cuda<<<blocksPerGrid, threadsPerBlock, smem_size>>>(d_img, d_img_copy);

#  else     // !CUDA_SHARED
    oil_cuda<<<blocksPerGrid, threadsPerBlock>>>(d_img, d_img_copy);

#  endif    // !CUDA_SHARED

#endif  // !CUDA_SECOND_VERSION

    SAFE_CUDA_CALL(cudaDeviceSynchronize());
    SAFE_CUDA_CALL(cudaMemcpy(h_img.ptr(), d_img, h_img_size, cudaMemcpyDeviceToHost));

#ifdef CUDA_SECOND_VERSION
    SAFE_CUDA_CALL(cudaFree(d_intensity));
#endif
    SAFE_CUDA_CALL(cudaFree(d_img_copy));
    SAFE_CUDA_CALL(cudaFree(d_img));
}
