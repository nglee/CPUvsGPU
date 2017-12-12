/*
 * Original cpu code is from "supercomputingblog.com".
 * (http://supercomputingblog.com/openmp/image-twist-and-swirl-algorithm/2/)
 *
 * Modified to work with OpenCV, OpenMP
 */

#include <stdio.h>
#include <chrono>

#include <opencv2/opencv.hpp>

#ifndef CUDA
#  include <omp.h>
#endif

#define C_PI 3.141592653589793238462643383279502884197169399375f

using namespace std;

#ifdef CUDA
extern void swirl_cuda_wrapper(cv::Mat& img, float factor);
#else
void swirl(cv::Mat& img, float factor)
{
    int width = img.cols;
    int height = img.rows;

    float cX = (width - 1) / 2.0f;
    float cY = (height - 1) / 2.0f;

    cv::Mat img_copy = img.clone();

    #pragma omp parallel for
    for (int i = 0; i < height; i++) {

        float relY = cY - i;
        for (int j = 0; j < width; j++) {

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

            img.at<cv::Vec3b>(i, j) = img_copy.at<cv::Vec3b>(srcY, srcX);
        }
    }
}
#endif

int main()
{
    for (int i = 0; i < 8; i++) {

        char path[100];
        snprintf(path, 100, "../images/%02d.jpg", i);

        cv::Mat h_img = cv::imread(path);

        auto start_clk = std::chrono::high_resolution_clock::now();

#ifdef CUDA
        swirl_cuda_wrapper(h_img, 2.f);
        snprintf(path, 100, "./%02d.twisted.cuda.jpg", i);
#else
        swirl(h_img, 2.f);
#  ifdef OMP
        snprintf(path, 100, "./%02d.twisted.omp.jpg", i);
#  else
        snprintf(path, 100, "./%02d.twisted.jpg", i);
#  endif
#endif

        auto end_clk = std::chrono::high_resolution_clock::now();

        printf("(%d) %10.4f ms - image size (%d, %d)\n", i,
               std::chrono::duration<float, std::milli>(end_clk - start_clk).count(),
               h_img.cols, h_img.rows);

        cv::imwrite(path, h_img);
    }

    return 0;
}
