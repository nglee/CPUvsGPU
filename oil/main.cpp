/* Algorithm and original cpu code are from "supercomputingblog.com".
 * (http://supercomputingblog.com/graphics/oil-painting-algorithm/)
 *
 * Modified to work with OpenCV
 */

#include <stdio.h>
#include <chrono>

#include <opencv2/opencv.hpp>

#ifndef CUDA
#  include <omp.h>
#endif

#define RADIUS          5
#define INTENSITY_LEVEL 20

using namespace std;

#define CROP(X) X>255 ? 255 : X<0 ? 0 : X

#ifdef CUDA
extern void oil_cuda_wrapper(cv::Mat& h_img, const int radius, const int intensity_level);
#else
void oil(cv::Mat& img, const int radius, const int intensity_level)
{
    int width = img.cols;
    int height = img.rows;

    cv::Mat img_copy = img.clone(); // deep copy

    #pragma omp parallel for
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++) {

            unsigned int buf[intensity_level * 4] = { 0 };
            unsigned int *intensity_count   = &buf[0];
            unsigned int *average_r         = &buf[intensity_level];
            unsigned int *average_g         = &buf[intensity_level * 2];
            unsigned int *average_b         = &buf[intensity_level * 3];

            //memset(buf, 0, sizeof(unsigned int) * intensity_level * 4);

            for (int k = i-radius; k <= i+radius; k++) {
                if (k<0 || k>=height)
                    continue;
                for (int l = j-radius; l <= j+radius; l++) {
                    if (l<0 || l>=width)
                        continue;

                    unsigned int r = img_copy.at<cv::Vec3b>(k, l)[0];
                    unsigned int g = img_copy.at<cv::Vec3b>(k, l)[1];
                    unsigned int b = img_copy.at<cv::Vec3b>(k, l)[2];

                    unsigned int cur_intensity = (r+g+b)*intensity_level/766;
                    intensity_count[cur_intensity]++;
                    average_r[cur_intensity] += r;
                    average_g[cur_intensity] += g;
                    average_b[cur_intensity] += b;

                    /*
                    if (i == 354 && j == 194)
                        printf("k = %d, l = %d, cur_intensity = %u\n", k, l, cur_intensity);
                    */

                }
            }

            unsigned int max = 0;
            unsigned int max_index = 0;

            for (int k = 0; k < intensity_level; k++)
                if (intensity_count[k] > max) {
                    max = intensity_count[k];
                    max_index = k;
                }
            /*
            if (max == 0) {
                printf("i = %d, j = %d, max = %u, max_index = %u\n", i, j, max, max_index);
                for (int k = 0; k < intensity_level; k++) {
                    printf("intensity_count[%d] = %u\n", k, intensity_count[k]);
                    printf("average_r[%d] = %u\n", k, average_r[k]);
                    printf("average_g[%d] = %u\n", k, average_g[k]);
                    printf("average_b[%d] = %u\n", k, average_b[k]);
                }
            }
            */

            unsigned int final_r = average_r[max_index] / max;
            unsigned int final_g = average_g[max_index] / max;
            unsigned int final_b = average_b[max_index] / max;

            img.at<cv::Vec3b>(i, j)[0] = CROP(final_r);
            img.at<cv::Vec3b>(i, j)[1] = CROP(final_g);
            img.at<cv::Vec3b>(i, j)[2] = CROP(final_b);
        }
}
#endif

int main()
{
    for (int i = 0; i < 8; i++) {

        char path[100];
        snprintf(path, 100, "../images/%02d.jpg", i);

        cv::Mat img = cv::imread(path);

        auto start_clk = std::chrono::high_resolution_clock::now();

#ifdef CUDA
        oil_cuda_wrapper(img, RADIUS, INTENSITY_LEVEL);
#  ifdef CUDA_SECOND_VERSION
        snprintf(path, 100, "./%02d.oil.cuda.second.jpg", i);
#  else  // !CUDA_SECOND_VERSION
#    ifdef CUDA_SHARED
        snprintf(path, 100, "./%02d.oil.cuda.shared.jpg", i);
#    else  // !CUDA_SHARED
        snprintf(path, 100, "./%02d.oil.cuda.jpg", i);
#    endif // !CUDA_SHARED
#  endif // !CUDA_SECOND_VERSION
#else  // !CUDA
        oil(img, RADIUS, INTENSITY_LEVEL);
#  ifdef OMP
        snprintf(path, 100, "./%02d.oil.omp.jpg", i);
#  else  // !OMP
        snprintf(path, 100, "./%02d.oil.jpg", i);
#  endif // !OMP
#endif // !CUDA

        auto end_clk = std::chrono::high_resolution_clock::now();

        printf("(%d) %10.4f ms - image size (%d, %d)\n", i,
               std::chrono::duration<float, std::milli>(end_clk - start_clk).count(),
               img.cols, img.rows);

        cv::imwrite(path, img);
    }

    return 0;
}
