#pragma once
#include <cstdint>

#pragma pack(push, 1)
struct BMPFileHeader
{
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
};

struct BMPDIBHeader
{
    uint32_t biSize;
    int32_t  biWidth;
    int32_t  biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t  biXPelsPerMeter;
    int32_t  biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
};
#pragma pack(pop)

namespace Bmp
{
    bool BmpToRgbBuffers(const char* filePath, unsigned char** rBuf, unsigned char** gBuf, unsigned char** bBuf, int& width, int& height);
    bool RgbBuffersToBmp(const char* outputPath, const unsigned char* rBuf, const unsigned char* gBuf, const unsigned char* bBuf, const int& width, const int& height);
}
