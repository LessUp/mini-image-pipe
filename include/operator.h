#pragma once

#include "types.h"
#include <cuda_runtime.h>
#include <string>
#include <memory>

namespace mini_image_pipe {

// Abstract base class for all image operators
class IOperator {
public:
    virtual ~IOperator() = default;

    // Execute operator on GPU stream
    virtual cudaError_t execute(
        const void* input,
        void* output,
        int width,
        int height,
        int channels,
        cudaStream_t stream
    ) = 0;

    // Get output dimensions given input dimensions
    virtual void getOutputDimensions(
        int inputWidth, int inputHeight, int inputChannels,
        int& outputWidth, int& outputHeight, int& outputChannels
    ) const = 0;

    // Get operator name for debugging
    virtual const char* getName() const = 0;

    // Get required output buffer size in bytes
    size_t getOutputBufferSize(int inputWidth, int inputHeight, int inputChannels) const {
        int outW, outH, outC;
        getOutputDimensions(inputWidth, inputHeight, inputChannels, outW, outH, outC);
        return static_cast<size_t>(outW) * outH * outC * sizeof(uint8_t);
    }
};

// Shared pointer type for operators
using OperatorPtr = std::shared_ptr<IOperator>;

} // namespace mini_image_pipe
