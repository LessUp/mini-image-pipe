#include "operators/gaussian_blur.h"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

namespace mini_image_pipe {

#define TILE_SIZE 16
#define MAX_KERNEL_RADIUS 3  // For 7x7 kernel

// Horizontal pass kernel with shared memory and halo regions
__global__ void gaussianHorizontalKernel(
    const uint8_t* input,
    float* output,
    int width,
    int height,
    int channels,
    const float* kernel,
    int kernelRadius
) {
    extern __shared__ float sharedMem[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    int sharedWidth = blockDim.x + 2 * kernelRadius;
    
    // Load tile with halo into shared memory
    for (int c = 0; c < channels; c++) {
        float* sharedChannel = sharedMem + c * sharedWidth * blockDim.y;
        
        // Load main tile
        int loadX = x;
        int loadY = y;
        if (loadX < width && loadY < height) {
            int idx = (loadY * width + loadX) * channels + c;
            sharedChannel[ty * sharedWidth + tx + kernelRadius] = static_cast<float>(input[idx]);
        } else {
            sharedChannel[ty * sharedWidth + tx + kernelRadius] = 0.0f;
        }

        // Load left halo
        if (tx < kernelRadius) {
            int haloX = x - kernelRadius;
            // Reflection padding
            if (haloX < 0) haloX = -haloX - 1;
            haloX = min(haloX, width - 1);
            
            if (y < height) {
                int idx = (y * width + haloX) * channels + c;
                sharedChannel[ty * sharedWidth + tx] = static_cast<float>(input[idx]);
            } else {
                sharedChannel[ty * sharedWidth + tx] = 0.0f;
            }
        }

        // Load right halo
        if (tx >= blockDim.x - kernelRadius) {
            int haloX = x + kernelRadius;
            // Reflection padding
            if (haloX >= width) haloX = 2 * width - haloX - 1;
            haloX = max(haloX, 0);
            
            if (y < height) {
                int idx = (y * width + haloX) * channels + c;
                sharedChannel[ty * sharedWidth + tx + 2 * kernelRadius] = static_cast<float>(input[idx]);
            } else {
                sharedChannel[ty * sharedWidth + tx + 2 * kernelRadius] = 0.0f;
            }
        }
    }

    __syncthreads();

    if (x >= width || y >= height) return;

    // Apply horizontal convolution
    for (int c = 0; c < channels; c++) {
        float* sharedChannel = sharedMem + c * sharedWidth * blockDim.y;
        float sum = 0.0f;
        
        for (int k = -kernelRadius; k <= kernelRadius; k++) {
            sum += sharedChannel[ty * sharedWidth + tx + kernelRadius + k] * kernel[k + kernelRadius];
        }
        
        output[(y * width + x) * channels + c] = sum;
    }
}

// Vertical pass kernel
__global__ void gaussianVerticalKernel(
    const float* input,
    uint8_t* output,
    int width,
    int height,
    int channels,
    const float* kernel,
    int kernelRadius
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        for (int k = -kernelRadius; k <= kernelRadius; k++) {
            int ny = y + k;
            // Reflection padding
            if (ny < 0) ny = -ny - 1;
            if (ny >= height) ny = 2 * height - ny - 1;
            ny = max(0, min(ny, height - 1));
            
            sum += input[(ny * width + x) * channels + c] * kernel[k + kernelRadius];
        }
        
        // Clamp and convert to uint8
        sum = fminf(fmaxf(sum, 0.0f), 255.0f);
        output[(y * width + x) * channels + c] = static_cast<uint8_t>(sum);
    }
}

// Simple single-pass Gaussian blur (fallback)
__global__ void gaussianBlurSimpleKernel(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height,
    int channels,
    const float* kernel,
    int kernelRadius
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int kernelSize = 2 * kernelRadius + 1;

    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        for (int ky = -kernelRadius; ky <= kernelRadius; ky++) {
            for (int kx = -kernelRadius; kx <= kernelRadius; kx++) {
                int nx = x + kx;
                int ny = y + ky;
                
                // Reflection padding
                if (nx < 0) nx = -nx - 1;
                if (nx >= width) nx = 2 * width - nx - 1;
                if (ny < 0) ny = -ny - 1;
                if (ny >= height) ny = 2 * height - ny - 1;
                
                nx = max(0, min(nx, width - 1));
                ny = max(0, min(ny, height - 1));
                
                float kernelVal = kernel[ky + kernelRadius] * kernel[kx + kernelRadius];
                sum += static_cast<float>(input[(ny * width + nx) * channels + c]) * kernelVal;
            }
        }
        
        sum = fminf(fmaxf(sum, 0.0f), 255.0f);
        output[(y * width + x) * channels + c] = static_cast<uint8_t>(sum);
    }
}

GaussianBlurOperator::GaussianBlurOperator(GaussianKernelSize size, float sigma)
    : kernelSize_(size)
    , sigma_(sigma) {
    generateKernel();
}

GaussianBlurOperator::~GaussianBlurOperator() {
    freeResources();
}

void GaussianBlurOperator::generateKernel() {
    int size = static_cast<int>(kernelSize_);
    int radius = size / 2;
    
    // Calculate sigma if not specified
    float s = sigma_;
    if (s <= 0.0f) {
        s = 0.3f * ((size - 1) * 0.5f - 1) + 0.8f;
    }
    
    // Generate 1D Gaussian kernel
    std::vector<float> kernel(size);
    float sum = 0.0f;
    
    for (int i = 0; i < size; i++) {
        float x = static_cast<float>(i - radius);
        kernel[i] = expf(-(x * x) / (2.0f * s * s));
        sum += kernel[i];
    }
    
    // Normalize
    for (int i = 0; i < size; i++) {
        kernel[i] /= sum;
    }
    
    // Free old kernel if exists
    if (d_kernel_) {
        cudaFree(d_kernel_);
    }
    
    // Allocate and copy to device
    cudaMalloc(&d_kernel_, size * sizeof(float));
    cudaMemcpy(d_kernel_, kernel.data(), size * sizeof(float), cudaMemcpyHostToDevice);
}

void GaussianBlurOperator::freeResources() {
    if (d_kernel_) {
        cudaFree(d_kernel_);
        d_kernel_ = nullptr;
    }
    if (d_intermediate_) {
        cudaFree(d_intermediate_);
        d_intermediate_ = nullptr;
        intermediateSize_ = 0;
    }
}

void GaussianBlurOperator::setKernelSize(GaussianKernelSize size) {
    if (kernelSize_ != size) {
        kernelSize_ = size;
        generateKernel();
    }
}

void GaussianBlurOperator::setSigma(float sigma) {
    if (sigma_ != sigma) {
        sigma_ = sigma;
        generateKernel();
    }
}

cudaError_t GaussianBlurOperator::execute(
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

    if (!d_kernel_) {
        return cudaErrorInvalidValue;
    }

    const uint8_t* inputPtr = static_cast<const uint8_t*>(input);
    uint8_t* outputPtr = static_cast<uint8_t*>(output);

    int kernelRadius = static_cast<int>(kernelSize_) / 2;

    // Allocate intermediate buffer if needed
    size_t requiredSize = static_cast<size_t>(width) * height * channels * sizeof(float);
    if (intermediateSize_ < requiredSize) {
        if (d_intermediate_) {
            cudaFree(d_intermediate_);
        }
        cudaError_t err = cudaMalloc(&d_intermediate_, requiredSize);
        if (err != cudaSuccess) {
            d_intermediate_ = nullptr;
            intermediateSize_ = 0;
            return err;
        }
        intermediateSize_ = requiredSize;
    }

    float* intermediatePtr = static_cast<float*>(d_intermediate_);

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    // Calculate shared memory size for horizontal pass
    int sharedWidth = blockSize.x + 2 * kernelRadius;
    size_t sharedMemSize = channels * sharedWidth * blockSize.y * sizeof(float);

    // Horizontal pass
    gaussianHorizontalKernel<<<gridSize, blockSize, sharedMemSize, stream>>>(
        inputPtr, intermediatePtr, width, height, channels, d_kernel_, kernelRadius
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    // Vertical pass
    gaussianVerticalKernel<<<gridSize, blockSize, 0, stream>>>(
        intermediatePtr, outputPtr, width, height, channels, d_kernel_, kernelRadius
    );

    return cudaGetLastError();
}

void GaussianBlurOperator::getOutputDimensions(
    int inputWidth, int inputHeight, int inputChannels,
    int& outputWidth, int& outputHeight, int& outputChannels
) const {
    outputWidth = inputWidth;
    outputHeight = inputHeight;
    outputChannels = inputChannels;
}

} // namespace mini_image_pipe
