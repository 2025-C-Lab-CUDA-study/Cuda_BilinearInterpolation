#include<cuda_runtime.h>
#include<device_launch_parameters.h>

#include <iostream>
#include"BmpUtile.h"


#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CUDA_KERNEL_CHECK() CUDA_CHECK(cudaGetLastError())


// Typedef ================

using uchar = unsigned char;


// Consts ================

constexpr int RATIO = 1;
constexpr int BLOCK = 16;


__global__ void BilinearReduce(uchar* dstBuffer, size_t dstPitch, uchar* srcBuffer, size_t srcPitch, int dstWidth, int dstHeight, int srcWidth, int srcHeight)
{

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;


    if (x < dstWidth && y < dstHeight)
    {
        int srcX = x * (srcWidth / dstWidth);
        int srcY = y * (srcHeight / dstHeight);

        int x1 = srcX;
        int x2 = min(srcX + RATIO, srcWidth - RATIO);
        int y1 = srcY;
        int y2 = min(srcY + RATIO, srcHeight - RATIO);

        uchar c1 = srcBuffer[x1 + srcPitch * y1];
        uchar c2 = srcBuffer[x1 + srcPitch * y2];
        uchar c3 = srcBuffer[x2 + srcPitch * y1];
        uchar c4 = srcBuffer[x2 + srcPitch * y2];
        float value = (c1 + c2 + c3 + c4) * 0.25;

        dstBuffer[x + dstPitch * y] = static_cast<uchar>(value);
    }
}

__global__ void BilinearIncrease(uchar* dstBuffer, size_t dstPitch, uchar* srcBuffer, size_t srcPitch, int dstWidth, int dstHeight, int srcWidth, int srcHeight)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < dstWidth && y < dstHeight)
    {
        int srcX = static_cast<int>(x * (static_cast<float>(srcWidth) / dstWidth));
        int srcY = static_cast<int>(y * (static_cast<float>(srcHeight) / dstHeight));

        int x1 = srcX;
        int x2 = min(srcX + RATIO, srcWidth - RATIO);
        int y1 = srcY;
        int y2 = min(srcY + RATIO, srcHeight - RATIO);

        uchar c1 = srcBuffer[x1 + srcPitch * y1];
        uchar c2 = srcBuffer[x1 + srcPitch * y2];
        uchar c3 = srcBuffer[x2 + srcPitch * y1];
        uchar c4 = srcBuffer[x2 + srcPitch * y2];
        float value = (c1 + c2 + c3 + c4) * 0.25;

        dstBuffer[x + dstPitch * y] = value;
    }
}


int main(void)
{

    // Set Host data =========================================================================================

    uchar* h_rb = nullptr, * h_rbCal = nullptr, * h_rrbb = nullptr;
    uchar* h_gb = nullptr, * h_gbCal = nullptr, * h_ggbb = nullptr;
    uchar* h_bb = nullptr, * h_bbCal = nullptr, * h_bbbb = nullptr;
    int h_width, h_height;

    const char* path = "C:\\Users\\james\\Documents\\2025\\source_code\\lenna.bmp";
    if (!Bmp::BmpToRgbBuffers(path, &h_rb, &h_gb, &h_bb, h_width, h_height))
    {
        if (!h_rb) free(h_rb);
        if (!h_gb) free(h_gb);
        if (!h_bb) free(h_bb);
    }

    h_rbCal = (uchar*)malloc(sizeof(uchar) * (h_width / 2) * (h_height / 2));
    h_gbCal = (uchar*)malloc(sizeof(uchar) * (h_width / 2) * (h_height / 2));
    h_bbCal = (uchar*)malloc(sizeof(uchar) * (h_width / 2) * (h_height / 2));

    h_rrbb = (uchar*)malloc(sizeof(uchar) * h_width * h_height);
    h_ggbb = (uchar*)malloc(sizeof(uchar) * h_width * h_height);
    h_bbbb = (uchar*)malloc(sizeof(uchar) * h_width * h_height);

    // Set Device data ========================================================================================

    int d_width = h_width;
    int d_height = h_height;
    size_t firstPitch, secondPitch;
    uchar* d_rb_first = nullptr, * d_rb_second = nullptr;
    uchar* d_gb_first = nullptr, * d_gb_second = nullptr;
    uchar* d_bb_first = nullptr, * d_bb_second = nullptr;

    CUDA_CHECK(cudaMallocPitch(&d_rb_first, &firstPitch, sizeof(uchar) * d_width, d_height));
    CUDA_CHECK(cudaMallocPitch(&d_gb_first, &firstPitch, sizeof(uchar) * d_width, d_height));
    CUDA_CHECK(cudaMallocPitch(&d_bb_first, &firstPitch, sizeof(uchar) * d_width, d_height));

    CUDA_CHECK(cudaMallocPitch(&d_rb_second, &secondPitch, sizeof(uchar) * d_width / 2, d_height / 2));
    CUDA_CHECK(cudaMallocPitch(&d_gb_second, &secondPitch, sizeof(uchar) * d_width / 2, d_height / 2));
    CUDA_CHECK(cudaMallocPitch(&d_bb_second, &secondPitch, sizeof(uchar) * d_width / 2, d_height / 2));

    CUDA_CHECK(cudaMemcpy2D(d_rb_first, firstPitch, h_rb, sizeof(uchar) * h_width, sizeof(char) * h_width, h_height, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(d_gb_first, firstPitch, h_gb, sizeof(uchar) * h_width, sizeof(char) * h_width, h_height, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(d_bb_first, firstPitch, h_bb, sizeof(uchar) * h_width, sizeof(char) * h_width, h_height, cudaMemcpyHostToDevice));


    // Run Bilinear Resize =====================================================================================

    dim3 block(BLOCK, BLOCK);
    dim3 grid(((d_width / 2) + BLOCK - 1) / BLOCK, ((d_height / 2) + BLOCK - 1) / BLOCK);


    BilinearReduce <<<grid, block >>> (d_rb_second, secondPitch, d_rb_first, firstPitch, d_width / 2, d_height / 2, d_width, d_height);
    CUDA_KERNEL_CHECK();
    BilinearReduce <<<grid, block >>> (d_gb_second, secondPitch, d_gb_first, firstPitch, d_width / 2, d_height / 2, d_width, d_height);
    CUDA_KERNEL_CHECK();
    BilinearReduce <<<grid, block >>> (d_bb_second, secondPitch, d_bb_first, firstPitch, d_width / 2, d_height / 2, d_width, d_height);
    CUDA_KERNEL_CHECK();

    dim3 grid2 = ((d_width + BLOCK - 1) / BLOCK, (d_height + BLOCK - 1) / BLOCK);

    BilinearIncrease << <grid2, block >> > (d_rb_first, firstPitch, d_rb_second, secondPitch, d_width, d_height, d_width / 2, d_height / 2);
    CUDA_KERNEL_CHECK();
    BilinearIncrease << <grid2, block >> > (d_gb_first, firstPitch, d_gb_second, secondPitch, d_width, d_height, d_width / 2, d_height / 2);
    CUDA_KERNEL_CHECK();
    BilinearIncrease << <grid2, block >> > (d_bb_first, firstPitch, d_bb_second, secondPitch, d_width, d_height, d_width / 2, d_height / 2);
    CUDA_KERNEL_CHECK();

    // Store resized image ======================================================================================

    CUDA_CHECK(cudaMemcpy2D(h_rbCal, sizeof(uchar) * (h_width / 2), d_rb_second, secondPitch, d_width / 2, d_height / 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(h_gbCal, sizeof(uchar) * (h_width / 2), d_gb_second, secondPitch, d_width / 2, d_height / 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(h_bbCal, sizeof(uchar) * (h_width / 2), d_bb_second, secondPitch, d_width / 2, d_height / 2, cudaMemcpyDeviceToHost));

    const char* outPath = "C:\\Users\\james\\Documents\\2025\\source_code\\reduceLenna.bmp";
    if (!Bmp::RgbBuffersToBmp(outPath, h_rbCal, h_gbCal, h_bbCal, h_width / 2, h_height / 2))
    {
        std::cout << "Error : Writing bmp file failed";
    }

    CUDA_CHECK(cudaMemcpy2D(h_rrbb, sizeof(uchar) * h_width, d_rb_first, firstPitch, d_width, d_height, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(h_ggbb, sizeof(uchar) * h_width, d_gb_first, firstPitch, d_width, d_height, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(h_bbbb, sizeof(uchar) * h_width, d_bb_first, firstPitch, d_width, d_height, cudaMemcpyDeviceToHost));

    const char* outPath2 = "C:\\Users\\james\\Documents\\2025\\source_code\\increasedLenna.bmp";
    if (!Bmp::RgbBuffersToBmp(outPath2, h_rrbb, h_ggbb, h_bbbb, h_width, h_height))
    {
        std::cout << "Error : Writing bmp file failed";
    }

    // free =====================================================================================================

    free(h_rb);
    free(h_gb);
    free(h_bb);
    free(h_rbCal);
    free(h_gbCal);
    free(h_bbCal);
    free(h_rrbb);
    free(h_ggbb);
    free(h_bbbb);
    cudaFree(d_rb_first);
    cudaFree(d_gb_first);
    cudaFree(d_bb_first);
    cudaFree(d_rb_second);
    cudaFree(d_gb_second);
    cudaFree(d_bb_second);

    return 0;
}

