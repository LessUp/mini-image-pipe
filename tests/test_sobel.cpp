#include <gtest/gtest.h>
#include "operators/sobel.h"
#include "memory_manager.h"
#include <random>
#include <cmath>
#include <cstring>

using namespace mini_image_pipe;

// Feature: mini-image-pipe, Property 4: Sobel Gradient Computation
// Validates: Requirements 2.1, 2.2
TEST(SobelPropertyTest, GradientComputation) {
    MemoryManager& mgr = MemoryManager::getInstance();
    SobelOperator op;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> sizeDist(16, 64);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Test with known edge patterns
    for (int iter = 0; iter < 100; iter++) {
        int width = sizeDist(gen);
        int height = sizeDist(gen);
        int channels = 1;  // Grayscale for easier verification
        
        size_t inputSize = width * height * channels;
        size_t outputSize = width * height;
        
        // Allocate buffers
        uint8_t* h_input = static_cast<uint8_t*>(mgr.allocatePinned(inputSize));
        uint8_t* h_output = static_cast<uint8_t*>(mgr.allocatePinned(outputSize));
        uint8_t* d_input = static_cast<uint8_t*>(mgr.allocateDevice(inputSize));
        uint8_t* d_output = static_cast<uint8_t*>(mgr.allocateDevice(outputSize));
        
        // Create a vertical edge pattern (left half dark, right half bright)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                h_input[y * width + x] = (x < width / 2) ? 0 : 255;
            }
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
        
        // Verify: vertical edge should produce high Gx values at the edge
        // The edge is at x = width/2
        int edgeX = width / 2;
        
        // Check that edge pixels have higher magnitude than non-edge pixels
        float edgeSum = 0;
        float nonEdgeSum = 0;
        int edgeCount = 0;
        int nonEdgeCount = 0;
        
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                float val = h_output[y * width + x];
                
                if (std::abs(x - edgeX) <= 1) {
                    edgeSum += val;
                    edgeCount++;
                } else if (std::abs(x - edgeX) > 3) {
                    nonEdgeSum += val;
                    nonEdgeCount++;
                }
            }
        }
        
        if (edgeCount > 0 && nonEdgeCount > 0) {
            float edgeAvg = edgeSum / edgeCount;
            float nonEdgeAvg = nonEdgeSum / nonEdgeCount;
            
            // Edge pixels should have higher average magnitude
            EXPECT_GT(edgeAvg, nonEdgeAvg) 
                << "Edge detection failed: edge avg=" << edgeAvg 
                << ", non-edge avg=" << nonEdgeAvg;
        }
        
        // Cleanup
        mgr.freePinned(h_input);
        mgr.freePinned(h_output);
        mgr.freeDevice(d_input);
        mgr.freeDevice(d_output);
    }
    
    cudaStreamDestroy(stream);
}

// Feature: mini-image-pipe, Property 5: Sobel Single-Channel Output
// Validates: Requirements 2.4
TEST(SobelPropertyTest, SingleChannelOutput) {
    MemoryManager& mgr = MemoryManager::getInstance();
    SobelOperator op;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> sizeDist(16, 64);
    std::uniform_int_distribution<int> channelDist(1, 4);
    std::uniform_int_distribution<uint8_t> pixelDist(0, 255);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    for (int iter = 0; iter < 100; iter++) {
        int width = sizeDist(gen);
        int height = sizeDist(gen);
        int inputChannels = channelDist(gen);
        
        // Verify output dimensions
        int outW, outH, outC;
        op.getOutputDimensions(width, height, inputChannels, outW, outH, outC);
        
        EXPECT_EQ(outW, width);
        EXPECT_EQ(outH, height);
        EXPECT_EQ(outC, 1) << "Sobel output should always be single-channel";
        
        size_t inputSize = width * height * inputChannels;
        size_t outputSize = width * height;  // Single channel
        
        // Allocate buffers
        uint8_t* h_input = static_cast<uint8_t*>(mgr.allocatePinned(inputSize));
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
        cudaError_t err = op.execute(d_input, d_output, width, height, inputChannels, stream);
        EXPECT_EQ(err, cudaSuccess);
        
        cudaStreamSynchronize(stream);
        
        // Cleanup
        mgr.freePinned(h_input);
        mgr.freeDevice(d_input);
        mgr.freeDevice(d_output);
    }
    
    cudaStreamDestroy(stream);
}
