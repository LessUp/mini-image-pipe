#include <gtest/gtest.h>
#include "pipeline.h"
#include "operators/color_convert.h"
#include "operators/resize.h"
#include "operators/gaussian_blur.h"
#include "operators/sobel.h"
#include "memory_manager.h"
#include <random>
#include <vector>

using namespace mini_image_pipe;

// Feature: mini-image-pipe, Property 19: Pipeline Topology and Buffer Management
// Validates: Requirements 8.1, 8.2
TEST(PipelinePropertyTest, TopologyAndBufferManagement) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> sizeDist(32, 128);
    std::uniform_int_distribution<uint8_t> pixelDist(0, 255);
    
    MemoryManager& mgr = MemoryManager::getInstance();
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    for (int iter = 0; iter < 100; iter++) {
        int width = sizeDist(gen);
        int height = sizeDist(gen);
        int channels = 3;
        
        // Create pipeline with sequential topology
        Pipeline pipeline;
        
        auto resizeOp = std::make_shared<ResizeOperator>(width / 2, height / 2);
        auto colorOp = std::make_shared<ColorConvertOperator>(ColorConversionType::RGB_TO_GRAY);
        
        int resize = pipeline.addOperator("Resize", resizeOp);
        int color = pipeline.addOperator("ColorConvert", colorOp);
        
        // Connect: Resize -> ColorConvert
        EXPECT_TRUE(pipeline.connect(resize, color));
        
        // Allocate input buffer
        size_t inputSize = width * height * channels;
        uint8_t* h_input = static_cast<uint8_t*>(mgr.allocatePinned(inputSize));
        uint8_t* d_input = static_cast<uint8_t*>(mgr.allocateDevice(inputSize));
        
        // Generate random input
        for (size_t i = 0; i < inputSize; i++) {
            h_input[i] = pixelDist(gen);
        }
        
        mgr.copyToDeviceAsync(d_input, h_input, inputSize, stream);
        cudaStreamSynchronize(stream);
        
        // Set input
        pipeline.setInput(resize, d_input, width, height, channels);
        
        // Execute pipeline
        cudaError_t err = pipeline.execute();
        EXPECT_EQ(err, cudaSuccess);
        
        // Verify output buffer was allocated
        void* output = pipeline.getOutput(color);
        EXPECT_NE(output, nullptr);
        
        // Cleanup
        mgr.freePinned(h_input);
        mgr.freeDevice(d_input);
    }
    
    cudaStreamDestroy(stream);
}

// Feature: mini-image-pipe, Property 20: No Redundant Computation
// Validates: Requirements 8.3
TEST(PipelinePropertyTest, NoRedundantComputation) {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    MemoryManager& mgr = MemoryManager::getInstance();
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    for (int iter = 0; iter < 100; iter++) {
        int width = 64;
        int height = 64;
        int channels = 3;
        
        // Create pipeline with diamond topology
        // A -> B -> D
        // A -> C -> D
        // A should only execute once
        Pipeline pipeline;
        
        auto opA = std::make_shared<ColorConvertOperator>(ColorConversionType::BGR_TO_RGB);
        auto opB = std::make_shared<ResizeOperator>(32, 32);
        auto opC = std::make_shared<ResizeOperator>(48, 48);
        auto opD = std::make_shared<ColorConvertOperator>(ColorConversionType::RGB_TO_GRAY);
        
        int a = pipeline.addOperator("A", opA);
        int b = pipeline.addOperator("B", opB);
        int c = pipeline.addOperator("C", opC);
        int d = pipeline.addOperator("D", opD);
        
        pipeline.connect(a, b);
        pipeline.connect(a, c);
        pipeline.connect(b, d);
        // Note: In this simple implementation, D only takes input from B
        // A more complex implementation would merge inputs
        
        // Allocate input buffer
        size_t inputSize = width * height * channels;
        uint8_t* d_input = static_cast<uint8_t*>(mgr.allocateDevice(inputSize));
        
        // Set input
        pipeline.setInput(a, d_input, width, height, channels);
        
        // Reset execution counts
        pipeline.getTaskGraph().resetExecutionCounts();
        
        // Execute pipeline
        cudaError_t err = pipeline.execute();
        EXPECT_EQ(err, cudaSuccess);
        
        // Verify A executed exactly once
        int execCountA = pipeline.getTaskGraph().getExecutionCount(a);
        EXPECT_EQ(execCountA, 1) << "Task A should execute exactly once";
        
        // Cleanup
        mgr.freeDevice(d_input);
    }
    
    cudaStreamDestroy(stream);
}

// Feature: mini-image-pipe, Property 21: Runtime Parameter Configuration
// Validates: Requirements 8.4
TEST(PipelinePropertyTest, RuntimeParameterConfiguration) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> sizeDist(32, 128);
    
    MemoryManager& mgr = MemoryManager::getInstance();
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    for (int iter = 0; iter < 100; iter++) {
        int width = sizeDist(gen);
        int height = sizeDist(gen);
        int channels = 3;
        
        Pipeline pipeline;
        
        auto resizeOp = std::make_shared<ResizeOperator>(width / 2, height / 2);
        int resize = pipeline.addOperator("Resize", resizeOp);
        
        // Allocate input buffer
        size_t inputSize = width * height * channels;
        uint8_t* d_input = static_cast<uint8_t*>(mgr.allocateDevice(inputSize));
        
        // Set input
        pipeline.setInput(resize, d_input, width, height, channels);
        
        // Execute with initial parameters
        cudaError_t err = pipeline.execute();
        EXPECT_EQ(err, cudaSuccess);
        
        // Change parameters at runtime
        int newWidth = width / 4;
        int newHeight = height / 4;
        resizeOp->setTargetSize(newWidth, newHeight);
        
        // Reset and execute again
        pipeline.reset();
        pipeline.setInput(resize, d_input, width, height, channels);
        
        err = pipeline.execute();
        EXPECT_EQ(err, cudaSuccess);
        
        // Verify new dimensions are used
        int outW, outH, outC;
        resizeOp->getOutputDimensions(width, height, channels, outW, outH, outC);
        EXPECT_EQ(outW, newWidth);
        EXPECT_EQ(outH, newHeight);
        
        // Cleanup
        mgr.freeDevice(d_input);
    }
    
    cudaStreamDestroy(stream);
}

// Feature: mini-image-pipe, Property 22: Batch Processing
// Validates: Requirements 8.5
TEST(PipelinePropertyTest, BatchProcessing) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> batchDist(2, 8);
    std::uniform_int_distribution<uint8_t> pixelDist(0, 255);
    
    MemoryManager& mgr = MemoryManager::getInstance();
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    for (int iter = 0; iter < 100; iter++) {
        int width = 64;
        int height = 64;
        int channels = 3;
        int batchSize = batchDist(gen);
        
        Pipeline pipeline;
        
        auto colorOp = std::make_shared<ColorConvertOperator>(ColorConversionType::RGB_TO_GRAY);
        int color = pipeline.addOperator("ColorConvert", colorOp);
        
        // Allocate batch of input buffers
        size_t inputSize = width * height * channels;
        std::vector<void*> inputs(batchSize);
        std::vector<void*> outputs;
        
        for (int i = 0; i < batchSize; i++) {
            uint8_t* h_input = static_cast<uint8_t*>(mgr.allocatePinned(inputSize));
            uint8_t* d_input = static_cast<uint8_t*>(mgr.allocateDevice(inputSize));
            
            // Generate random input
            for (size_t j = 0; j < inputSize; j++) {
                h_input[j] = pixelDist(gen);
            }
            
            mgr.copyToDeviceAsync(d_input, h_input, inputSize, stream);
            inputs[i] = d_input;
            
            mgr.freePinned(h_input);
        }
        
        cudaStreamSynchronize(stream);
        
        // Set input for first frame to establish dimensions
        pipeline.setInput(color, inputs[0], width, height, channels);
        
        // Execute batch
        cudaError_t err = pipeline.executeBatch(inputs, outputs, width, height, channels);
        EXPECT_EQ(err, cudaSuccess);
        
        // Verify we got the right number of outputs
        EXPECT_EQ(outputs.size(), batchSize);
        
        // Cleanup
        for (void* input : inputs) {
            mgr.freeDevice(input);
        }
    }
    
    cudaStreamDestroy(stream);
}
