#include "operators/resize.h"
#include <cuda_runtime.h>
#include <cmath>

namespace mini_image_pipe {

// CUDA kernel for nearest-neighbor interpolation
__global__ void resizeNearestKernel(
    const uint8_t* input,
    uint8_t* output,
    int srcWidth,
    int srcHeight,
    int dstWidth,
    int dstHeight,
    int channels
) {
    int dstX = blockIdx.x * blockDim.x + threadIdx.x;
    int dstY = blockIdx.y * blockDim.y + threadIdx.y;

    if (dstX >= dstWidth || dstY >= dstHeight) return;

    // Compute source coordinates
    float scaleX = static_cast<float>(srcWidth) / dstWidth;
    float scaleY = static_cast<float>(srcHeight) / dstHeight;

    int srcX = static_cast<int>(dstX * scaleX);
    int srcY = static_cast<int>(dstY * scaleY);

    // Clamp to valid range
    srcX = min(max(srcX, 0), srcWidth - 1);
    srcY = min(max(srcY, 0), srcHeight - 1);

    int srcIdx = (srcY * srcWidth + srcX) * channels;
    int dstIdx = (dstY * dstWidth + dstX) * channels;

    for (int c = 0; c < channels; c++) {
        output[dstIdx + c] = input[srcIdx + c];
    }
}

// CUDA kernel for bilinear interpolation
__global__ void resizeBilinearKernel(
    const uint8_t* input,
    uint8_t* output,
    int srcWidth,
    int srcHeight,
    int dstWidth,
    int dstHeight,
    int channels
) {
    int dstX = blockIdx.x * blockDim.x + threadIdx.x;
    int dstY = blockIdx.y * blockDim.y + threadIdx.y;

    if (dstX >= dstWidth || dstY >= dstHeight) return;

    // Compute source coordinates (center-aligned)
    float scaleX = static_cast<float>(srcWidth) / dstWidth;
    float scaleY = static_cast<float>(srcHeight) / dstHeight;

    float srcXf = (dstX + 0.5f) * scaleX - 0.5f;
    float srcYf = (dstY + 0.5f) * scaleY - 0.5f;

    // Get integer and fractional parts
    int x0 = static_cast<int>(floorf(srcXf));
    int y0 = static_cast<int>(floorf(srcYf));
    float fx = srcXf - x0;
    float fy = srcYf - y0;

    // Clamp coordinates
    int x1 = min(x0 + 1, srcWidth - 1);
    int y1 = min(y0 + 1, srcHeight - 1);
    x0 = max(x0, 0);
    y0 = max(y0, 0);

    int dstIdx = (dstY * dstWidth + dstX) * channels;

    for (int c = 0; c < channels; c++) {
        // Get four neighboring pixels
        float p00 = static_cast<float>(input[(y0 * srcWidth + x0) * channels + c]);
        float p10 = static_cast<float>(input[(y0 * srcWidth + x1) * channels + c]);
        float p01 = static_cast<float>(input[(y1 * srcWidth + x0) * channels + c]);
        float p11 = static_cast<float>(input[(y1 * srcWidth + x1) * channels + c]);

        // Bilinear interpolation
        float top = p00 * (1.0f - fx) + p10 * fx;
        float bottom = p01 * (1.0f - fx) + p11 * fx;
        float value = top * (1.0f - fy) + bottom * fy;

        output[dstIdx + c] = static_cast<uint8_t>(min(max(value, 0.0f), 255.0f));
    }
}

ResizeOperator::ResizeOperator(int targetWidth, int targetHeight, InterpolationMode mode)
    : targetWidth_(targetWidth)
    , targetHeight_(targetHeight)
    , mode_(mode) {}

void ResizeOperator::setTargetSize(int width, int height) {
    targetWidth_ = width;
    targetHeight_ = height;
}

void ResizeOperator::setInterpolationMode(InterpolationMode mode) {
    mode_ = mode;
}

cudaError_t ResizeOperator::execute(
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

    if (targetWidth_ <= 0 || targetHeight_ <= 0) {
        return cudaErrorInvalidValue;
    }

    const uint8_t* inputPtr = static_cast<const uint8_t*>(input);
    uint8_t* outputPtr = static_cast<uint8_t*>(output);

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (targetWidth_ + blockSize.x - 1) / blockSize.x,
        (targetHeight_ + blockSize.y - 1) / blockSize.y
    );

    switch (mode_) {
        case InterpolationMode::NEAREST:
            resizeNearestKernel<<<gridSize, blockSize, 0, stream>>>(
                inputPtr, outputPtr,
                width, height,
                targetWidth_, targetHeight_,
                channels
            );
            break;

        case InterpolationMode::BILINEAR:
            resizeBilinearKernel<<<gridSize, blockSize, 0, stream>>>(
                inputPtr, outputPtr,
                width, height,
                targetWidth_, targetHeight_,
                channels
            );
            break;

        default:
            return cudaErrorInvalidValue;
    }

    return cudaGetLastError();
}

void ResizeOperator::getOutputDimensions(
    int inputWidth, int inputHeight, int inputChannels,
    int& outputWidth, int& outputHeight, int& outputChannels
) const {
    outputWidth = targetWidth_;
    outputHeight = targetHeight_;
    outputChannels = inputChannels;
}

} // namespace mini_image_pipe
