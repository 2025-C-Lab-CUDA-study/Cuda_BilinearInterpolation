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
    int src_pitch = srcPitch / sizeof(uchar);
    int dst_pitch = dstPitch / sizeof(uchar);

    for (
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        y < dstHeight;
        y += blockDim.y * gridDim.y
        )
    {
        for (
            int x = blockDim.x * blockIdx.x + threadIdx.x;
            x < dstWidth;
            x += blockDim.x * gridDim.x
            )
        {
            int srcX = min(max(x * 2, 0), srcWidth - 1);
            int srcY = min(max(y * 2, 0), srcHeight - 1);

            uchar c1 = srcBuffer[srcX     + src_pitch * srcY];
            uchar c2 = srcBuffer[srcX + 1 + src_pitch * srcY];
            uchar c3 = srcBuffer[srcX     + src_pitch * (srcY + 1)];
            uchar c4 = srcBuffer[srcX + 1 + src_pitch * (srcY + 1)];

            dstBuffer[x + dst_pitch * y] = static_cast<uchar>((c1 + c2 + c3 + c4) * 0.25);
        }
    }
}

__global__ void BilinearIncrease(uchar* dstBuffer, size_t dstPitch, uchar* srcBuffer, size_t srcPitch, int dstWidth, int dstHeight, int srcWidth, int srcHeight)
{
    int src_pitch = srcPitch / sizeof(uchar);
    int dst_pitch = dstPitch / sizeof(uchar);
    
    for (
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        y < dstHeight;
        y += blockDim.y * gridDim.y
        )
    {
        for (
            int x = blockDim.x * blockIdx.x + threadIdx.x;
            x < dstWidth;
            x += blockDim.x * gridDim.x
            )
        { 
            int srcX = min(max(x / 2, 0), dstWidth - 1);
            int srcY = min(max(y / 2, 0), dstHeight - 1);

            uchar c1 = srcBuffer[srcX + src_pitch * srcY];
            uchar c2 = srcBuffer[srcX + 1 + src_pitch * srcY];
            uchar c3 = srcBuffer[srcX + src_pitch * (srcY + 1)];
            uchar c4 = srcBuffer[srcX + 1 + src_pitch * (srcY + 1)];

            dstBuffer[x + dst_pitch * y] = static_cast<uchar>((c1 + c2 + c3 + c4) * 0.25);
        }
    }
}




int main(void)
{

    // Set Host data =========================================================================================

    uchar* h_rb = nullptr;
    uchar* h_gb = nullptr;
    uchar* h_bb = nullptr;
    int h_width, h_height;

    const char* path = "C:\\Users\\james\\Documents\\2025\\source_code\\lenna.bmp";
    if (!Bmp::BmpToRgbBuffers(path, &h_rb, &h_gb, &h_bb, h_width, h_height))
    {
        if (!h_rb) free(h_rb);
        if (!h_gb) free(h_gb);
        if (!h_bb) free(h_bb);
    }

    // Set Device data ========================================================================================

    int d_width = h_width;
    int d_height = h_height;
    size_t firstPitch, secondPitch, calPitch;
    uchar* d_rb_first = nullptr, * d_rb_second = nullptr;
    uchar* d_gb_first = nullptr, * d_gb_second = nullptr;
    uchar* d_bb_first = nullptr, * d_bb_second = nullptr;

    CUDA_CHECK(cudaMallocPitch(&d_rb_first, &firstPitch, sizeof(uchar) * d_width, d_height)); // fisrt buffer
    CUDA_CHECK(cudaMallocPitch(&d_gb_first, &firstPitch, sizeof(uchar) * d_width, d_height));
    CUDA_CHECK(cudaMallocPitch(&d_bb_first, &firstPitch, sizeof(uchar) * d_width, d_height));

    CUDA_CHECK(cudaMallocPitch(&d_rb_second, &secondPitch, sizeof(uchar) * (d_width), d_height)); // second buffer
    CUDA_CHECK(cudaMallocPitch(&d_gb_second, &secondPitch, sizeof(uchar) * (d_width), d_height));
    CUDA_CHECK(cudaMallocPitch(&d_bb_second, &secondPitch, sizeof(uchar) * (d_width), d_height));

    CUDA_CHECK(cudaMemcpy2D(d_rb_first, firstPitch, h_rb, sizeof(uchar) * h_width, sizeof(char) * h_width, h_height, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(d_gb_first, firstPitch, h_gb, sizeof(uchar) * h_width, sizeof(char) * h_width, h_height, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(d_bb_first, firstPitch, h_bb, sizeof(uchar) * h_width, sizeof(char) * h_width, h_height, cudaMemcpyHostToDevice));


    // Run Bilinear Resize =====================================================================================

    int offset = 1;
    int transWidth = d_width / 4;
    int transHeight = d_height / 4;

    dim3 block(BLOCK, BLOCK);
    dim3 grid((transWidth + BLOCK - 1) / BLOCK, (transHeight + BLOCK - 1) / BLOCK);

    BilinearReduce <<<grid, block>>> (d_rb_second, secondPitch, d_rb_first, firstPitch, d_width / 2, d_height / 2, d_width, d_height);
    CUDA_KERNEL_CHECK();
    BilinearReduce <<<grid, block>>> (d_gb_second, secondPitch, d_gb_first, firstPitch, d_width / 2, d_height / 2, d_width, d_height);
    CUDA_KERNEL_CHECK();
    BilinearReduce <<<grid, block>>> (d_bb_second, secondPitch, d_bb_first, firstPitch, d_width / 2, d_height / 2, d_width, d_height);
    CUDA_KERNEL_CHECK();

    transWidth /= 2;
    transHeight /= 2;

    CUDA_CHECK(cudaMemcpy2D(h_rb, sizeof(uchar) * h_width, d_rb_second, secondPitch, d_width / 2, d_height / 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(h_gb, sizeof(uchar) * h_width, d_gb_second, secondPitch, d_width / 2, d_height / 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(h_bb, sizeof(uchar) * h_width, d_bb_second, secondPitch, d_width / 2, d_height / 2, cudaMemcpyDeviceToHost));

    const char* outPath3 = "C:\\Users\\james\\Documents\\2025\\source_code\\middleCheckLenna.bmp";
    if (!Bmp::RgbBuffersToBmp(outPath3, h_rb, h_gb, h_bb, h_width, h_height))
    {
        std::cout << "Error : Writing bmp file failed";
    }


    dim3 grid2((transWidth + BLOCK - 1) / BLOCK, (transHeight + BLOCK - 1) / BLOCK);

    BilinearIncrease <<<grid2, block>>> (d_rb_first, firstPitch, d_rb_second, secondPitch, d_width, d_height, d_width / 2, d_height / 2);
    CUDA_KERNEL_CHECK();
    BilinearIncrease <<<grid2, block>>> (d_gb_first, firstPitch, d_gb_second, secondPitch, d_width, d_height, d_width / 2, d_height / 2);
    CUDA_KERNEL_CHECK();
    BilinearIncrease <<<grid2, block>>> (d_bb_first, firstPitch, d_bb_second, secondPitch, d_width, d_height, d_width / 2, d_height / 2);
    CUDA_KERNEL_CHECK();


    // Store resized image ======================================================================================

    CUDA_CHECK(cudaMemcpy2D(h_rb, sizeof(uchar) * h_width, d_rb_first, firstPitch, d_width, d_height, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(h_gb, sizeof(uchar) * h_width, d_gb_first, firstPitch, d_width, d_height, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(h_bb, sizeof(uchar) * h_width, d_bb_first, firstPitch, d_width, d_height, cudaMemcpyDeviceToHost));

    const char* outPath2 = "C:\\Users\\james\\Documents\\2025\\source_code\\kawaseBluredLenna.bmp";
    if (!Bmp::RgbBuffersToBmp(outPath2, h_rb, h_gb, h_bb, h_width, h_height))
    {
        std::cout << "Error : Writing bmp file failed";
    }

    // free =====================================================================================================

    free(h_rb);
    free(h_gb);
    free(h_bb);
    cudaFree(d_rb_first);
    cudaFree(d_gb_first);
    cudaFree(d_bb_first);
    cudaFree(d_rb_second);
    cudaFree(d_gb_second);
    cudaFree(d_bb_second);

    return 0;
}

