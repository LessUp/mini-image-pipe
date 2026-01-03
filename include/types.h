#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace mini_image_pipe {

// Image buffer structure
struct ImageBuffer {
    void* data = nullptr;       // Pointer to pixel data
    int width = 0;              // Image width in pixels
    int height = 0;             // Image height in pixels
    int channels = 0;           // Number of channels (1, 3, or 4)
    int stride = 0;             // Row stride in bytes
    bool isDeviceMemory = false;// True if data is on GPU
    bool isPinned = false;      // True if host memory is pinned

    size_t sizeInBytes() const {
        return static_cast<size_t>(stride) * height;
    }

    size_t pixelCount() const {
        return static_cast<size_t>(width) * height;
    }

    bool isValid() const {
        return data != nullptr && width > 0 && height > 0 && 
               channels > 0 && stride >= width * channels;
    }
};

// CUDA kernel configuration
struct KernelConfig {
    dim3 blockSize;   // Thread block dimensions
    dim3 gridSize;    // Grid dimensions
    size_t sharedMem = 0; // Shared memory size in bytes

    static KernelConfig forImage(int width, int height, int tileSize = 16) {
        KernelConfig cfg;
        cfg.blockSize = dim3(tileSize, tileSize);
        cfg.gridSize = dim3(
            (width + tileSize - 1) / tileSize,
            (height + tileSize - 1) / tileSize
        );
        cfg.sharedMem = 0;
        return cfg;
    }

    static KernelConfig forImage1D(int width, int height, int blockSize = 256) {
        KernelConfig cfg;
        int totalPixels = width * height;
        cfg.blockSize = dim3(blockSize);
        cfg.gridSize = dim3((totalPixels + blockSize - 1) / blockSize);
        cfg.sharedMem = 0;
        return cfg;
    }
};

// Pipeline configuration
struct PipelineConfig {
    int numStreams = 4;                          // Number of CUDA streams
    size_t pinnedPoolSize = 64 * 1024 * 1024;    // 64MB pinned memory pool
    bool enableProfiling = false;                // Enable CUDA profiling
    int maxBatchSize = 8;                        // Maximum frames in batch
};

// Color conversion types
enum class ColorConversionType {
    RGB_TO_GRAY,
    BGR_TO_RGB,
    RGBA_TO_RGB,
    GRAY_TO_RGB
};

// Interpolation modes for resize
enum class InterpolationMode {
    NEAREST,
    BILINEAR
};

// Gaussian blur kernel sizes
enum class GaussianKernelSize {
    KERNEL_3x3 = 3,
    KERNEL_5x5 = 5,
    KERNEL_7x7 = 7
};

// Task states for scheduler
enum class TaskState {
    PENDING,
    READY,
    RUNNING,
    COMPLETED,
    FAILED
};

// Error codes
enum class PipelineError {
    SUCCESS = 0,
    INVALID_INPUT,
    MEMORY_ALLOCATION_FAILED,
    CUDA_ERROR,
    CYCLE_DETECTED,
    TASK_FAILED,
    INVALID_CONFIGURATION
};

} // namespace mini_image_pipe
