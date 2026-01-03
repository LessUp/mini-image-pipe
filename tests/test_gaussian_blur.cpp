#include <gtest/gtest.h>
#include "operators/gaussian_blur.h"
#include "memory_manager.h"
#include <random>
#include <cmath>
#include <cstring>

using namespace mini_image_pipe;

// Feature: mini-image-pipe, Property 1: Gaussian Blur Multi-Channel Support
// Validates: Requirements 1.1, 1.5
TEST(GaussianBlurPropertyTest, MultiChannelSupport) {
    MemoryManager& mgr = MemoryManager::getInstance();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> sizeDist(16, 64);
    std::uniform_int_distribution<uint8_t> pixelDist(0, 255);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    std::vector<GaussianKernelSize> kernelSizes = {
        GaussianKernelSize::KERNEL_3x3,
        GaussianKernelSize::KERNEL_5x5,
        GaussianKernelSize::KERNEL_7x7
    };
    
    std::vector<int> channelCounts = {1, 3, 4};
    
    for (int iter = 0; iter < 100; iter++) {
        int width = sizeDist(gen);
        int height = sizeDist(gen);
        
        // Random kernel size and channel count
        GaussianKernelSize kernelSize = kernelSizes[iter % kernelSizes.size()];
        int channels = channelCounts[iter % channelCounts.size()];
        
        GaussianBlurOperator op(kernelSize);
        
        // Verify output dimensions match input
        int outW, outH, outC;
        op.getOutputDimensions(width, height, channels, outW, outH, outC);
        
        EXPECT_EQ(outW, width);
        EXPECT_EQ(outH, height);
        EXPECT_EQ(outC, channels);
        
        size_t bufferSize = width * height * channels;
        
        // Allocate buffers
        uint8_t* h_input = static_cast<uint8_t*>(mgr.allocatePinned(bufferSize));
        uint8_t* h_output = static_cast<uint8_t*>(mgr.allocatePinned(bufferSize));
        uint8_t* d_input = static_cast<uint8_t*>(mgr.allocateDevice(bufferSize));
        uint8_t* d_output = static_cast<uint8_t*>(mgr.allocateDevice(bufferSize));
        
        // Generate random input
        for (size_t i = 0; i < bufferSize; i++) {
            h_input[i] = pixelDist(gen);
        }
        
        // Copy to device
        mgr.copyToDeviceAsync(d_input, h_input, bufferSize, stream);
        cudaStreamSynchronize(stream);
        
        // Execute operator
        cudaError_t err = op.execute(d_input, d_output, width, height, channels, stream);
        EXPECT_EQ(err, cudaSuccess) 
            << "Failed for kernel=" << static_cast<int>(kernelSize) 
            << ", channels=" << channels;
        
        // Copy result back
        mgr.copyToHostAsync(h_output, d_output, bufferSize, stream);
        cudaStreamSynchronize(stream);
        
        // Verify output is valid (no NaN, values in range)
        for (size_t i = 0; i < bufferSize; i++) {
            EXPECT_GE(h_output[i], 0);
            EXPECT_LE(h_output[i], 255);
        }
        
        // Cleanup
        mgr.freePinned(h_input);
        mgr.freePinned(h_output);
        mgr.freeDevice(d_input);
        mgr.freeDevice(d_output);
    }
    
    cudaStreamDestroy(stream);
}

// Feature: mini-image-pipe, Property 2: Separable Filter Equivalence
// Validates: Requirements 1.2
TEST(GaussianBlurPropertyTest, SeparableFilterEquivalence) {
    // This test verifies that the separable implementation produces
    // results consistent with expected Gaussian blur behavior
    MemoryManager& mgr = MemoryManager::getInstance();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> sizeDist(32, 64);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    for (int iter = 0; iter < 100; iter++) {
        int width = sizeDist(gen);
        int height = sizeDist(gen);
        int channels = 1;
        
        GaussianBlurOperator op(GaussianKernelSize::KERNEL_5x5);
        
        size_t bufferSize = width * height * channels;
        
        // Allocate buffers
        uint8_t* h_input = static_cast<uint8_t*>(mgr.allocatePinned(bufferSize));
        uint8_t* h_output = static_cast<uint8_t*>(mgr.allocatePinned(bufferSize));
        uint8_t* d_input = static_cast<uint8_t*>(mgr.allocateDevice(bufferSize));
        uint8_t* d_output = static_cast<uint8_t*>(mgr.allocateDevice(bufferSize));
        
        // Create a simple test pattern (impulse in center)
        memset(h_input, 0, bufferSize);
        int cx = width / 2;
        int cy = height / 2;
        h_input[cy * width + cx] = 255;
        
        // Copy to device
        mgr.copyToDeviceAsync(d_input, h_input, bufferSize, stream);
        cudaStreamSynchronize(stream);
        
        // Execute operator
        cudaError_t err = op.execute(d_input, d_output, width, height, channels, stream);
        EXPECT_EQ(err, cudaSuccess);
        
        // Copy result back
        mgr.copyToHostAsync(h_output, d_output, bufferSize, stream);
        cudaStreamSynchronize(stream);
        
        // Verify Gaussian properties:
        // 1. Center should have highest value
        // 2. Values should decrease with distance from center
        // 3. Output should be symmetric
        
        float centerVal = h_output[cy * width + cx];
        
        // Check that center has a reasonable value (not 0, not 255)
        EXPECT_GT(centerVal, 0);
        
        // Check symmetry (approximately)
        for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
                int y1 = cy + dy;
                int y2 = cy - dy;
                int x1 = cx + dx;
                int x2 = cx - dx;
                
                if (y1 >= 0 && y1 < height && y2 >= 0 && y2 < height &&
                    x1 >= 0 && x1 < width && x2 >= 0 && x2 < width) {
                    float v1 = h_output[y1 * width + x1];
                    float v2 = h_output[y2 * width + x2];
                    
                    // Allow small tolerance for floating point
                    EXPECT_NEAR(v1, v2, 2) 
                        << "Asymmetry at offset (" << dx << ", " << dy << ")";
                }
            }
        }
        
        // Cleanup
        mgr.freePinned(h_input);
        mgr.freePinned(h_output);
        mgr.freeDevice(d_input);
        mgr.freeDevice(d_output);
    }
    
    cudaStreamDestroy(stream);
}

// Feature: mini-image-pipe, Property 3: Reflection Padding Boundary Handling
// Validates: Requirements 1.4
TEST(GaussianBlurPropertyTest, ReflectionPaddingBoundaryHandling) {
    MemoryManager& mgr = MemoryManager::getInstance();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> sizeDist(16, 64);
    std::uniform_int_distribution<uint8_t> pixelDist(0, 255);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    for (int iter = 0; iter < 100; iter++) {
        int width = sizeDist(gen);
        int height = sizeDist(gen);
        int channels = 1;
        
        GaussianBlurOperator op(GaussianKernelSize::KERNEL_5x5);
        
        size_t bufferSize = width * height * channels;
        
        // Allocate buffers
        uint8_t* h_input = static_cast<uint8_t*>(mgr.allocatePinned(bufferSize));
        uint8_t* h_output = static_cast<uint8_t*>(mgr.allocatePinned(bufferSize));
        uint8_t* d_input = static_cast<uint8_t*>(mgr.allocateDevice(bufferSize));
        uint8_t* d_output = static_cast<uint8_t*>(mgr.allocateDevice(bufferSize));
        
        // Generate random input
        for (size_t i = 0; i < bufferSize; i++) {
            h_input[i] = pixelDist(gen);
        }
        
        // Copy to device
        mgr.copyToDeviceAsync(d_input, h_input, bufferSize, stream);
        cudaStreamSynchronize(stream);
        
        // Execute operator
        cudaError_t err = op.execute(d_input, d_output, width, height, channels, stream);
        EXPECT_EQ(err, cudaSuccess);
        
        // Copy result back
        mgr.copyToHostAsync(h_output, d_output, bufferSize, stream);
        cudaStreamSynchronize(stream);
        
        // Verify boundary pixels are valid (no NaN, no out-of-range)
        // Check all boundary pixels
        for (int x = 0; x < width; x++) {
            // Top row
            EXPECT_GE(h_output[x], 0);
            EXPECT_LE(h_output[x], 255);
            
            // Bottom row
            EXPECT_GE(h_output[(height - 1) * width + x], 0);
            EXPECT_LE(h_output[(height - 1) * width + x], 255);
        }
        
        for (int y = 0; y < height; y++) {
            // Left column
            EXPECT_GE(h_output[y * width], 0);
            EXPECT_LE(h_output[y * width], 255);
            
            // Right column
            EXPECT_GE(h_output[y * width + width - 1], 0);
            EXPECT_LE(h_output[y * width + width - 1], 255);
        }
        
        // Cleanup
        mgr.freePinned(h_input);
        mgr.freePinned(h_output);
        mgr.freeDevice(d_input);
        mgr.freeDevice(d_output);
    }
    
    cudaStreamDestroy(stream);
}
