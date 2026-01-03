#include "operators/sobel.h"
#include <cuda_runtime.h>
#include <cmath>

namespace mini_image_pipe {

// Sobel kernels in constant memory
__constant__ int sobelGx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
__constant__ int sobelGy[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

// Tile size for shared memory
#define TILE_SIZE 16
#define BLOCK_SIZE (TILE_SIZE + 2)  // +2 for halo (1 pixel on each side)

// CUDA kernel for Sobel edge detection with shared memory
__global__ void sobelKernel(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height,
    int channels
) {
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_SIZE + tx;
    int y = blockIdx.y * TILE_SIZE + ty;

    // Load tile with halo into shared memory
    // Each thread loads one pixel, plus some threads load halo pixels
    int loadX = x - 1;
    int loadY = y - 1;

    // Clamp coordinates for boundary handling
    int clampedX = min(max(loadX, 0), width - 1);
    int clampedY = min(max(loadY, 0), height - 1);

    // Convert to grayscale if multi-channel, or use directly if single channel
    float pixelValue;
    if (channels == 1) {
        pixelValue = static_cast<float>(input[clampedY * width + clampedX]);
    } else {
        int idx = (clampedY * width + clampedX) * channels;
        float r = static_cast<float>(input[idx]);
        float g = static_cast<float>(input[idx + 1]);
        float b = static_cast<float>(input[idx + 2]);
        pixelValue = 0.299f * r + 0.587f * g + 0.114f * b;
    }

    if (tx < BLOCK_SIZE && ty < BLOCK_SIZE) {
        tile[ty][tx] = pixelValue;
    }

    // Load additional halo pixels for threads at the edge
    if (tx == TILE_SIZE - 1 && tx + 1 < BLOCK_SIZE) {
        int haloX = min(x + 1, width - 1);
        int haloY = min(max(y - 1, 0), height - 1);
        if (channels == 1) {
            tile[ty][tx + 2] = static_cast<float>(input[haloY * width + haloX]);
        } else {
            int idx = (haloY * width + haloX) * channels;
            float r = static_cast<float>(input[idx]);
            float g = static_cast<float>(input[idx + 1]);
            float b = static_cast<float>(input[idx + 2]);
            tile[ty][tx + 2] = 0.299f * r + 0.587f * g + 0.114f * b;
        }
    }

    if (ty == TILE_SIZE - 1 && ty + 1 < BLOCK_SIZE) {
        int haloX = min(max(x - 1, 0), width - 1);
        int haloY = min(y + 1, height - 1);
        if (channels == 1) {
            tile[ty + 2][tx] = static_cast<float>(input[haloY * width + haloX]);
        } else {
            int idx = (haloY * width + haloX) * channels;
            float r = static_cast<float>(input[idx]);
            float g = static_cast<float>(input[idx + 1]);
            float b = static_cast<float>(input[idx + 2]);
            tile[ty + 2][tx] = 0.299f * r + 0.587f * g + 0.114f * b;
        }
    }

    __syncthreads();

    // Only process pixels within the image bounds
    if (x >= width || y >= height) return;

    // Apply Sobel kernels
    float gx = 0.0f;
    float gy = 0.0f;

    // Sobel Gx kernel: [-1 0 1; -2 0 2; -1 0 1]
    // Sobel Gy kernel: [-1 -2 -1; 0 0 0; 1 2 1]
    for (int ky = 0; ky < 3; ky++) {
        for (int kx = 0; kx < 3; kx++) {
            int sx = tx + kx;
            int sy = ty + ky;
            if (sx < BLOCK_SIZE && sy < BLOCK_SIZE) {
                float val = tile[sy][sx];
                gx += val * sobelGx[ky * 3 + kx];
                gy += val * sobelGy[ky * 3 + kx];
            }
        }
    }

    // Compute gradient magnitude: sqrt(Gx² + Gy²)
    float magnitude = sqrtf(gx * gx + gy * gy);

    // Clamp to [0, 255]
    magnitude = fminf(fmaxf(magnitude, 0.0f), 255.0f);

    output[y * width + x] = static_cast<uint8_t>(magnitude);
}

// Simple Sobel kernel without shared memory (fallback)
__global__ void sobelKernelSimple(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height,
    int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float gx = 0.0f;
    float gy = 0.0f;

    // Sobel Gx kernel: [-1 0 1; -2 0 2; -1 0 1]
    // Sobel Gy kernel: [-1 -2 -1; 0 0 0; 1 2 1]
    const int sobelGxLocal[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    const int sobelGyLocal[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int nx = min(max(x + kx, 0), width - 1);
            int ny = min(max(y + ky, 0), height - 1);

            float pixelValue;
            if (channels == 1) {
                pixelValue = static_cast<float>(input[ny * width + nx]);
            } else {
                int idx = (ny * width + nx) * channels;
                float r = static_cast<float>(input[idx]);
                float g = static_cast<float>(input[idx + 1]);
                float b = static_cast<float>(input[idx + 2]);
                pixelValue = 0.299f * r + 0.587f * g + 0.114f * b;
            }

            int kidx = (ky + 1) * 3 + (kx + 1);
            gx += pixelValue * sobelGxLocal[kidx];
            gy += pixelValue * sobelGyLocal[kidx];
        }
    }

    // Compute gradient magnitude: sqrt(Gx² + Gy²)
    float magnitude = sqrtf(gx * gx + gy * gy);
    magnitude = fminf(fmaxf(magnitude, 0.0f), 255.0f);

    output[y * width + x] = static_cast<uint8_t>(magnitude);
}

SobelOperator::SobelOperator() {}

cudaError_t SobelOperator::execute(
    const void* input,
    void* output,
    int width,
    int height,
    int channels,
    cudaStream_t stream
) {
    if (!input || !output || width <= 0 || height <= 0 || channels <= 0) {
        return cudaErrorInvalidValue;
    }

    const uint8_t* inputPtr = static_cast<const uint8_t*>(input);
    uint8_t* outputPtr = static_cast<uint8_t*>(output);

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    // Use simple kernel for reliability
    sobelKernelSimple<<<gridSize, blockSize, 0, stream>>>(
        inputPtr, outputPtr, width, height, channels
    );

    return cudaGetLastError();
}

void SobelOperator::getOutputDimensions(
    int inputWidth, int inputHeight, int inputChannels,
    int& outputWidth, int& outputHeight, int& outputChannels
) const {
    outputWidth = inputWidth;
    outputHeight = inputHeight;
    outputChannels = 1;  // Always output single-channel gradient magnitude
}

} // namespace mini_image_pipe
