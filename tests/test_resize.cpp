#include <gtest/gtest.h>
#include "operators/resize.h"
#include "memory_manager.h"
#include <random>
#include <cmath>
#include <algorithm>

using namespace mini_image_pipe;

// Feature: mini-image-pipe, Property 6: Resize Coordinate Mapping
// Validates: Requirements 3.1, 3.2, 3.3
TEST(ResizePropertyTest, CoordinateMapping) {
    MemoryManager& mgr = MemoryManager::getInstance();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> sizeDist(32, 128);
    std::uniform_int_distribution<uint8_t> pixelDist(0, 255);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    for (int iter = 0; iter < 100; iter++) {
        int srcWidth = sizeDist(gen);
        int srcHeight = sizeDist(gen);
        int dstWidth = sizeDist(gen);
        int dstHeight = sizeDist(gen);
        int channels = 3;
        
        ResizeOperator op(dstWidth, dstHeight, InterpolationMode::NEAREST);
        
        size_t srcSize = srcWidth * srcHeight * channels;
        size_t dstSize = dstWidth * dstHeight * channels;
        
        // Allocate buffers
        uint8_t* h_input = static_cast<uint8_t*>(mgr.allocatePinned(srcSize));
        uint8_t* h_output = static_cast<uint8_t*>(mgr.allocatePinned(dstSize));
        uint8_t* d_input = static_cast<uint8_t*>(mgr.allocateDevice(srcSize));
        uint8_t* d_output = static_cast<uint8_t*>(mgr.allocateDevice(dstSize));
        
        // Generate random input
        for (size_t i = 0; i < srcSize; i++) {
            h_input[i] = pixelDist(gen);
        }
        
        // Copy to device
        mgr.copyToDeviceAsync(d_input, h_input, srcSize, stream);
        cudaStreamSynchronize(stream);
        
        // Execute operator
        cudaError_t err = op.execute(d_input, d_output, srcWidth, srcHeight, channels, stream);
        EXPECT_EQ(err, cudaSuccess);
        
        // Copy result back
        mgr.copyToHostAsync(h_output, d_output, dstSize, stream);
        cudaStreamSynchronize(stream);
        
        // Verify coordinate mapping for nearest neighbor
        float scaleX = static_cast<float>(srcWidth) / dstWidth;
        float scaleY = static_cast<float>(srcHeight) / dstHeight;
        
        for (int dstY = 0; dstY < dstHeight; dstY++) {
            for (int dstX = 0; dstX < dstWidth; dstX++) {
                int srcX = static_cast<int>(dstX * scaleX);
                int srcY = static_cast<int>(dstY * scaleY);
                srcX = std::min(std::max(srcX, 0), srcWidth - 1);
                srcY = std::min(std::max(srcY, 0), srcHeight - 1);
                
                int srcIdx = (srcY * srcWidth + srcX) * channels;
                int dstIdx = (dstY * dstWidth + dstX) * channels;
                
                for (int c = 0; c < channels; c++) {
                    EXPECT_EQ(h_output[dstIdx + c], h_input[srcIdx + c])
                        << "Mismatch at dst(" << dstX << ", " << dstY << ") channel " << c;
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

// Feature: mini-image-pipe, Property 7: Resize Arbitrary Scale Factors
// Validates: Requirements 3.4
TEST(ResizePropertyTest, ArbitraryScaleFactors) {
    MemoryManager& mgr = MemoryManager::getInstance();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> sizeDist(32, 128);
    std::uniform_real_distribution<float> scaleDist(0.25f, 4.0f);
    std::uniform_int_distribution<uint8_t> pixelDist(0, 255);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    for (int iter = 0; iter < 100; iter++) {
        int srcWidth = sizeDist(gen);
        int srcHeight = sizeDist(gen);
        float scaleX = scaleDist(gen);
        float scaleY = scaleDist(gen);
        
        int dstWidth = static_cast<int>(srcWidth * scaleX);
        int dstHeight = static_cast<int>(srcHeight * scaleY);
        
        // Ensure valid dimensions
        dstWidth = std::max(dstWidth, 1);
        dstHeight = std::max(dstHeight, 1);
        
        int channels = 3;
        
        ResizeOperator op(dstWidth, dstHeight, InterpolationMode::BILINEAR);
        
        // Verify output dimensions
        int outW, outH, outC;
        op.getOutputDimensions(srcWidth, srcHeight, channels, outW, outH, outC);
        
        EXPECT_EQ(outW, dstWidth);
        EXPECT_EQ(outH, dstHeight);
        EXPECT_EQ(outC, channels);
        
        size_t srcSize = srcWidth * srcHeight * channels;
        size_t dstSize = dstWidth * dstHeight * channels;
        
        // Allocate buffers
        uint8_t* h_input = static_cast<uint8_t*>(mgr.allocatePinned(srcSize));
        uint8_t* d_input = static_cast<uint8_t*>(mgr.allocateDevice(srcSize));
        uint8_t* d_output = static_cast<uint8_t*>(mgr.allocateDevice(dstSize));
        
        // Generate random input
        for (size_t i = 0; i < srcSize; i++) {
            h_input[i] = pixelDist(gen);
        }
        
        // Copy to device
        mgr.copyToDeviceAsync(d_input, h_input, srcSize, stream);
        cudaStreamSynchronize(stream);
        
        // Execute operator - should succeed for any scale factor
        cudaError_t err = op.execute(d_input, d_output, srcWidth, srcHeight, channels, stream);
        EXPECT_EQ(err, cudaSuccess) 
            << "Failed for scale (" << scaleX << ", " << scaleY << ")";
        
        cudaStreamSynchronize(stream);
        
        // Cleanup
        mgr.freePinned(h_input);
        mgr.freeDevice(d_input);
        mgr.freeDevice(d_output);
    }
    
    cudaStreamDestroy(stream);
}
