#pragma once

#include "operator.h"
#include "types.h"

namespace mini_image_pipe {

class SobelOperator : public IOperator {
public:
    SobelOperator();
    ~SobelOperator() override = default;

    cudaError_t execute(
        const void* input,
        void* output,
        int width,
        int height,
        int channels,
        cudaStream_t stream
    ) override;

    void getOutputDimensions(
        int inputWidth, int inputHeight, int inputChannels,
        int& outputWidth, int& outputHeight, int& outputChannels
    ) const override;

    const char* getName() const override { return "Sobel"; }
};

} // namespace mini_image_pipe
