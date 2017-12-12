#include <stdio.h>
#include <chrono>

#include <opencv2/opencv.hpp>

#ifdef CUDA
#  include <opencv2/gpu/gpu.hpp>
#endif

int main()
{
    for (int i = 0; i < 8; i++) {

        char path[100];
        snprintf(path, 100, "../images/%02d.jpg", i);

        cv::Mat h_img = cv::imread(path);
        cv::Mat h_resized;

#ifdef CUDA
        cv::gpu::GpuMat d_src, d_dst;

        d_src.upload(h_img);

        auto start_clk = std::chrono::high_resolution_clock::now();
        cv::gpu::resize(d_src, d_dst, cv::Size(), 1.2, 1.2, CV_INTER_CUBIC);
        auto end_clk = std::chrono::high_resolution_clock::now();

        d_dst.download(h_resized);

        printf("(%d) %10.4f ms - (%d, %d) resized to (%d, %d)\n", i,
               std::chrono::duration<float, std::milli>(end_clk - start_clk).count(),
               d_src.cols, d_src.rows, d_dst.cols, d_dst.rows);

        snprintf(path, 100, "./%02d.resized.gpu.jpg", i);
#else
        auto start_clk = std::chrono::high_resolution_clock::now();
        cv::resize(h_img, h_resized, cv::Size(), 1.2, 1.2, CV_INTER_CUBIC);
        auto end_clk = std::chrono::high_resolution_clock::now();

        printf("(%d) %10.4f ms - (%d, %d) resized to (%d, %d)\n", i,
               std::chrono::duration<float, std::milli>(end_clk - start_clk).count(),
               h_img.cols, h_img.rows, h_resized.cols, h_resized.rows);

        snprintf(path, 100, "./%02d.resized.cpu.jpg", i);
#endif

        cv::imwrite(path, h_resized);
    }

    return 0;
}
