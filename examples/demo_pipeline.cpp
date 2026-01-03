#include "pipeline.h"
#include "operators/resize.h"
#include "operators/color_convert.h"
#include "operators/gaussian_blur.h"
#include "operators/sobel.h"
#include "memory_manager.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>

using namespace mini_image_pipe;

// Generate a simple test image with a gradient pattern
void generateTestImage(uint8_t* data, int width, int height, int channels) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * channels;
            
            // Create a gradient pattern
            uint8_t r = static_cast<uint8_t>((x * 255) / width);
            uint8_t g = static_cast<uint8_t>((y * 255) / height);
            uint8_t b = static_cast<uint8_t>(((x + y) * 255) / (width + height));
            
            data[idx] = r;
            if (channels >= 2) data[idx + 1] = g;
            if (channels >= 3) data[idx + 2] = b;
            if (channels >= 4) data[idx + 3] = 255; // Alpha
        }
    }
}

int main() {
    std::cout << "Mini-ImagePipe Demo Pipeline" << std::endl;
    std::cout << "=============================" << std::endl;

    // Configuration
    const int inputWidth = 640;
    const int inputHeight = 480;
    const int inputChannels = 3;
    const int targetWidth = 320;
    const int targetHeight = 240;

    // Create pipeline
    PipelineConfig config;
    config.numStreams = 4;
    Pipeline pipeline(config);

    // Create operators
    auto resizeOp = std::make_shared<ResizeOperator>(
        targetWidth, targetHeight, InterpolationMode::BILINEAR
    );
    auto colorConvertOp = std::make_shared<ColorConvertOperator>(
        ColorConversionType::RGB_TO_GRAY
    );
    auto gaussianOp = std::make_shared<GaussianBlurOperator>(
        GaussianKernelSize::KERNEL_5x5
    );
    auto sobelOp = std::make_shared<SobelOperator>();

    // Add operators to pipeline
    // Pipeline: Resize -> ColorConvert -> GaussianBlur -> Sobel
    int resizeNode = pipeline.addOperator("Resize", resizeOp);
    int colorNode = pipeline.addOperator("ColorConvert", colorConvertOp);
    int blurNode = pipeline.addOperator("GaussianBlur", gaussianOp);
    int sobelNode = pipeline.addOperator("Sobel", sobelOp);

    // Connect operators
    pipeline.connect(resizeNode, colorNode);
    pipeline.connect(colorNode, blurNode);
    pipeline.connect(blurNode, sobelNode);

    std::cout << "Pipeline created with " << pipeline.getTaskGraph().size() << " operators" << std::endl;
    std::cout << "  - Resize: " << inputWidth << "x" << inputHeight 
              << " -> " << targetWidth << "x" << targetHeight << std::endl;
    std::cout << "  - ColorConvert: RGB -> Grayscale" << std::endl;
    std::cout << "  - GaussianBlur: 5x5 kernel" << std::endl;
    std::cout << "  - Sobel: Edge detection" << std::endl;

    // Allocate input image on host
    size_t inputSize = inputWidth * inputHeight * inputChannels;
    MemoryManager& memMgr = MemoryManager::getInstance();
    
    uint8_t* h_input = static_cast<uint8_t*>(memMgr.allocatePinned(inputSize));
    if (!h_input) {
        std::cerr << "Failed to allocate input buffer" << std::endl;
        return 1;
    }

    // Generate test image
    generateTestImage(h_input, inputWidth, inputHeight, inputChannels);
    std::cout << "Generated test image: " << inputWidth << "x" << inputHeight 
              << "x" << inputChannels << std::endl;

    // Allocate device memory for input
    uint8_t* d_input = static_cast<uint8_t*>(memMgr.allocateDevice(inputSize));
    if (!d_input) {
        std::cerr << "Failed to allocate device input buffer" << std::endl;
        memMgr.freePinned(h_input);
        return 1;
    }

    // Copy input to device
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    memMgr.copyToDeviceAsync(d_input, h_input, inputSize, stream);
    cudaStreamSynchronize(stream);

    // Set input for the pipeline
    pipeline.setInput(resizeNode, d_input, inputWidth, inputHeight, inputChannels);

    // Execute pipeline
    std::cout << "Executing pipeline..." << std::endl;
    cudaError_t err = pipeline.execute();

    if (err != cudaSuccess) {
        std::cerr << "Pipeline execution failed: " << cudaGetErrorString(err) << std::endl;
        memMgr.freeDevice(d_input);
        memMgr.freePinned(h_input);
        cudaStreamDestroy(stream);
        return 1;
    }

    std::cout << "Pipeline executed successfully!" << std::endl;

    // Get output
    void* d_output = pipeline.getOutput(sobelNode);
    if (d_output) {
        // Output is single-channel edge map
        size_t outputSize = targetWidth * targetHeight;
        uint8_t* h_output = static_cast<uint8_t*>(memMgr.allocatePinned(outputSize));
        
        if (h_output) {
            memMgr.copyToHostAsync(h_output, d_output, outputSize, stream);
            cudaStreamSynchronize(stream);

            // Print some statistics about the output
            int nonZeroCount = 0;
            int maxVal = 0;
            for (size_t i = 0; i < outputSize; i++) {
                if (h_output[i] > 0) nonZeroCount++;
                if (h_output[i] > maxVal) maxVal = h_output[i];
            }

            std::cout << "Output statistics:" << std::endl;
            std::cout << "  - Size: " << targetWidth << "x" << targetHeight << std::endl;
            std::cout << "  - Non-zero pixels: " << nonZeroCount 
                      << " (" << (100.0 * nonZeroCount / outputSize) << "%)" << std::endl;
            std::cout << "  - Max edge value: " << maxVal << std::endl;

            memMgr.freePinned(h_output);
        }
    }

    // Cleanup
    memMgr.freeDevice(d_input);
    memMgr.freePinned(h_input);
    cudaStreamDestroy(stream);

    std::cout << "Demo completed successfully!" << std::endl;
    return 0;
}
