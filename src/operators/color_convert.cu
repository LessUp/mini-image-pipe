#include "operators/color_convert.h"
#include <cuda_runtime.h>

namespace mini_image_pipe {

// CUDA kernel for RGB to Grayscale conversion
__global__ void rgbToGrayKernel(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height,
    int inputChannels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pixelIdx = y * width + x;
    int inputIdx = pixelIdx * inputChannels;

    float r = static_cast<float>(input[inputIdx]);
    float g = static_cast<float>(input[inputIdx + 1]);
    float b = static_cast<float>(input[inputIdx + 2]);

    // Y = 0.299*R + 0.587*G + 0.114*B
    float gray = 0.299f * r + 0.587f * g + 0.114f * b;
    output[pixelIdx] = static_cast<uint8_t>(min(max(gray, 0.0f), 255.0f));
}

// CUDA kernel for BGR to RGB conversion
__global__ void bgrToRgbKernel(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height,
    int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pixelIdx = y * width + x;
    int idx = pixelIdx * channels;

    // Swap B and R channels
    output[idx] = input[idx + 2];     // R = B
    output[idx + 1] = input[idx + 1]; // G = G
    output[idx + 2] = input[idx];     // B = R

    // Preserve alpha if present
    if (channels == 4) {
        output[idx + 3] = input[idx + 3];
    }
}

// CUDA kernel for RGBA to RGB conversion
__global__ void rgbaToRgbKernel(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pixelIdx = y * width + x;
    int inputIdx = pixelIdx * 4;
    int outputIdx = pixelIdx * 3;

    output[outputIdx] = input[inputIdx];
    output[outputIdx + 1] = input[inputIdx + 1];
    output[outputIdx + 2] = input[inputIdx + 2];
}

// CUDA kernel for Grayscale to RGB conversion
__global__ void grayToRgbKernel(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pixelIdx = y * width + x;
    int outputIdx = pixelIdx * 3;

    uint8_t gray = input[pixelIdx];
    output[outputIdx] = gray;
    output[outputIdx + 1] = gray;
    output[outputIdx + 2] = gray;
}

ColorConvertOperator::ColorConvertOperator(ColorConversionType type)
    : type_(type) {}

cudaError_t ColorConvertOperator::execute(
    const void* input,
    void* output,
    int width,
    int height,
    int channels,
    cudaStream_t stream
) {
    if (!input || !output || width <= 0 || height <= 0) {
        return cudaErrorInvalidValue;
    }

    const uint8_t* inputPtr = static_cast<const uint8_t*>(input);
    uint8_t* outputPtr = static_cast<uint8_t*>(output);

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    switch (type_) {
        case ColorConversionType::RGB_TO_GRAY:
            if (channels < 3) return cudaErrorInvalidValue;
            rgbToGrayKernel<<<gridSize, blockSize, 0, stream>>>(
                inputPtr, outputPtr, width, height, channels
            );
            break;

        case ColorConversionType::BGR_TO_RGB:
            if (channels < 3) return cudaErrorInvalidValue;
            bgrToRgbKernel<<<gridSize, blockSize, 0, stream>>>(
                inputPtr, outputPtr, width, height, channels
            );
            break;

        case ColorConversionType::RGBA_TO_RGB:
            if (channels != 4) return cudaErrorInvalidValue;
            rgbaToRgbKernel<<<gridSize, blockSize, 0, stream>>>(
                inputPtr, outputPtr, width, height
            );
            break;

        case ColorConversionType::GRAY_TO_RGB:
            if (channels != 1) return cudaErrorInvalidValue;
            grayToRgbKernel<<<gridSize, blockSize, 0, stream>>>(
                inputPtr, outputPtr, width, height
            );
            break;

        default:
            return cudaErrorInvalidValue;
    }

    return cudaGetLastError();
}

void ColorConvertOperator::getOutputDimensions(
    int inputWidth, int inputHeight, int inputChannels,
    int& outputWidth, int& outputHeight, int& outputChannels
) const {
    outputWidth = inputWidth;
    outputHeight = inputHeight;

    switch (type_) {
        case ColorConversionType::RGB_TO_GRAY:
            outputChannels = 1;
            break;
        case ColorConversionType::BGR_TO_RGB:
            outputChannels = inputChannels; // Preserve channel count (3 or 4)
            break;
        case ColorConversionType::RGBA_TO_RGB:
            outputChannels = 3;
            break;
        case ColorConversionType::GRAY_TO_RGB:
            outputChannels = 3;
            break;
        default:
            outputChannels = inputChannels;
    }
}

} // namespace mini_image_pipe
