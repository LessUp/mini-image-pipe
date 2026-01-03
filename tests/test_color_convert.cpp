#include <gtest/gtest.h>
#include "operators/color_convert.h"
#include "memory_manager.h"
#include <random>
#include <cmath>
#include <cstring>

using namespace mini_image_pipe;

// Feature: mini-image-pipe, Property 8: RGB to Grayscale Formula
// Validates: Requirements 4.2
TEST(ColorConvertPropertyTest, RGBToGrayscaleFormula) {
    MemoryManager& mgr = MemoryManager::getInstance();
    ColorConvertOperator op(ColorConversionType::RGB_TO_GRAY);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> sizeDist(10, 100);
    std::uniform_int_distribution<uint8_t> pixelDist(0, 255);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    for (int iter = 0; iter < 100; iter++) {
        int width = sizeDist(gen);
        int height = sizeDist(gen);
        int channels = 3;
        
        size_t inputSize = width * height * channels;
        size_t outputSize = width * height;
        
        // Allocate buffers
        uint8_t* h_input = static_cast<uint8_t*>(mgr.allocatePinned(inputSize));
        uint8_t* h_output = static_cast<uint8_t*>(mgr.allocatePinned(outputSize));
        uint8_t* d_input = static_cast<uint8_t*>(mgr.allocateDevice(inputSize));
        uint8_t* d_output = static_cast<uint8_t*>(mgr.allocateDevice(outputSize));
        
        // Generate random input
        for (size_t i = 0; i < inputSize; i++) {
            h_input[i] = pixelDist(gen);
        }
        
        // Copy to device
        mgr.copyToDeviceAsync(d_input, h_input, inputSize, stream);
        cudaStreamSynchronize(stream);
        
        // Execute operator
        cudaError_t err = op.execute(d_input, d_output, width, height, channels, stream);
        EXPECT_EQ(err, cudaSuccess);
        
        // Copy result back
        mgr.copyToHostAsync(h_output, d_output, outputSize, stream);
        cudaStreamSynchronize(stream);
        
        // Verify formula: Y = 0.299*R + 0.587*G + 0.114*B
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = (y * width + x) * channels;
                float r = h_input[idx];
                float g = h_input[idx + 1];
                float b = h_input[idx + 2];
                
                float expected = 0.299f * r + 0.587f * g + 0.114f * b;
                uint8_t expectedByte = static_cast<uint8_t>(std::min(std::max(expected, 0.0f), 255.0f));
                
                int outIdx = y * width + x;
                EXPECT_NEAR(h_output[outIdx], expectedByte, 1) 
                    << "Mismatch at (" << x << ", " << y << ")";
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

// Feature: mini-image-pipe, Property 9: BGR to RGB Channel Swap
// Validates: Requirements 4.3
TEST(ColorConvertPropertyTest, BGRToRGBChannelSwap) {
    MemoryManager& mgr = MemoryManager::getInstance();
    ColorConvertOperator op(ColorConversionType::BGR_TO_RGB);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> sizeDist(10, 100);
    std::uniform_int_distribution<uint8_t> pixelDist(0, 255);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    for (int iter = 0; iter < 100; iter++) {
        int width = sizeDist(gen);
        int height = sizeDist(gen);
        int channels = 3;
        
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
        
        // Verify channel swap: output[0] = input[2], output[2] = input[0]
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = (y * width + x) * channels;
                
                EXPECT_EQ(h_output[idx], h_input[idx + 2])     // R = B
                    << "R channel mismatch at (" << x << ", " << y << ")";
                EXPECT_EQ(h_output[idx + 1], h_input[idx + 1]) // G = G
                    << "G channel mismatch at (" << x << ", " << y << ")";
                EXPECT_EQ(h_output[idx + 2], h_input[idx])     // B = R
                    << "B channel mismatch at (" << x << ", " << y << ")";
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

// Feature: mini-image-pipe, Property 10: Alpha Channel Preservation
// Validates: Requirements 4.4
TEST(ColorConvertPropertyTest, AlphaChannelPreservation) {
    MemoryManager& mgr = MemoryManager::getInstance();
    ColorConvertOperator op(ColorConversionType::BGR_TO_RGB);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> sizeDist(10, 100);
    std::uniform_int_distribution<uint8_t> pixelDist(0, 255);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    for (int iter = 0; iter < 100; iter++) {
        int width = sizeDist(gen);
        int height = sizeDist(gen);
        int channels = 4;  // RGBA/BGRA
        
        size_t bufferSize = width * height * channels;
        
        // Allocate buffers
        uint8_t* h_input = static_cast<uint8_t*>(mgr.allocatePinned(bufferSize));
        uint8_t* h_output = static_cast<uint8_t*>(mgr.allocatePinned(bufferSize));
        uint8_t* d_input = static_cast<uint8_t*>(mgr.allocateDevice(bufferSize));
        uint8_t* d_output = static_cast<uint8_t*>(mgr.allocateDevice(bufferSize));
        
        // Generate random input with random alpha values
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
        
        // Verify alpha channel is preserved
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = (y * width + x) * channels;
                
                EXPECT_EQ(h_output[idx + 3], h_input[idx + 3])
                    << "Alpha channel not preserved at (" << x << ", " << y << ")";
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
