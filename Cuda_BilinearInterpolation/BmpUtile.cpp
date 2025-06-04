#include "BmpUtile.h"
#include <iostream>
#include <fstream>
#include <vector>

namespace Bmp
{

    bool BmpToRgbBuffers(const char* filePath, unsigned char** rBuf, unsigned char** gBuf, unsigned char** bBuf, int& width, int& height)
    {
        std::ifstream file(filePath, std::ios::binary);
        if (!file)
        {
            std::cout << "Error: Failed to open file!" << std::endl;
            return false;
        }

        BMPFileHeader fileHeader;
        file.read(reinterpret_cast<char*>(&fileHeader), sizeof(fileHeader));
        if (fileHeader.bfType != 0x4D42) // 'BM'
        {
            std::cout << "Error: Not a BMP file!" << std::endl;
            return false;
        }

        BMPDIBHeader dibHeader;
        file.read(reinterpret_cast<char*>(&dibHeader), sizeof(dibHeader));

        width = dibHeader.biWidth;
        height = dibHeader.biHeight;

        std::cout << "Width: " << width << ", Height: " << height << ", bpp: " << dibHeader.biBitCount << std::endl;

        if (dibHeader.biBitCount != 24)
        {
            std::cout << "Error: Only 24bpp BMP supported!" << std::endl;
            return false;
        }

        size_t bufferSize = width * height;
        *rBuf = (unsigned char*)malloc(bufferSize);
        *gBuf = (unsigned char*)malloc(bufferSize);
        *bBuf = (unsigned char*)malloc(bufferSize);

        if (!*rBuf || !*gBuf || !*bBuf)
        {
            std::cout << "Error: Memory allocation failed!" << std::endl;
            if (*rBuf) free(*rBuf);
            if (*gBuf) free(*gBuf);
            if (*bBuf) free(*bBuf);
            return false;
        }

        int row_padded = (width * 3 + 3) & (~3);
        std::vector<unsigned char> row(row_padded);

        for (int y = 0; y < height; ++y)
        {
            file.read(reinterpret_cast<char*>(row.data()), row_padded);
            for (int x = 0; x < width; ++x)
            {
                size_t idx = (height - 1 - y) * width + x;
                (*bBuf)[idx] = row[x * 3 + 0];
                (*gBuf)[idx] = row[x * 3 + 1];
                (*rBuf)[idx] = row[x * 3 + 2];
            }
        }

        file.close();
        return true;
    }

    bool RgbBuffersToBmp(const char* outputPath, const unsigned char* rBuf, const unsigned char* gBuf, const unsigned char* bBuf, const int& width, const int& height)
    {
        std::ofstream file(outputPath, std::ios::binary);
        if (!file)
        {
            std::cout << "Error: Failed to open output file!" << std::endl;
            return false;
        }

        int row_padded = (width * 3 + 3) & (~3);
        int fileSize = 14 + 40 + row_padded * height;

        BMPFileHeader fileHeader = { 0 };
        fileHeader.bfType = 0x4D42;
        fileHeader.bfSize = fileSize;
        fileHeader.bfOffBits = 14 + 40;

        BMPDIBHeader dibHeader = { 0 };
        dibHeader.biSize = 40;
        dibHeader.biWidth = width;
        dibHeader.biHeight = height;
        dibHeader.biPlanes = 1;
        dibHeader.biBitCount = 24;
        dibHeader.biCompression = 0;
        dibHeader.biSizeImage = row_padded * height;

        file.write(reinterpret_cast<const char*>(&fileHeader), sizeof(fileHeader));
        file.write(reinterpret_cast<const char*>(&dibHeader), sizeof(dibHeader));

        std::vector<unsigned char> row(row_padded);
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                size_t idx = (height - 1 - y) * width + x;
                row[x * 3 + 0] = bBuf[idx];
                row[x * 3 + 1] = gBuf[idx];
                row[x * 3 + 2] = rBuf[idx];
            }
            for (int p = width * 3; p < row_padded; ++p)
                row[p] = 0;

            file.write(reinterpret_cast<const char*>(row.data()), row_padded);
        }

        file.close();
        return true;
    }

} // namespace Bmp
