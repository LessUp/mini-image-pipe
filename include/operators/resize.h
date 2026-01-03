#pragma once

#include "operator.h"
#include "types.h"

namespace mini_image_pipe {

class ResizeOperator : public IOperator {
public:
    ResizeOperator(int targetWidth, int targetHeight, InterpolationMode mode = InterpolationMode::BILINEAR);
    ~ResizeOperator() override = default;

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

    const char* getName() const override { return "Resize"; }

    // Setters for runtime configuration
    void setTargetSize(int width, int height);
    void setInterpolationMode(InterpolationMode mode);

    int getTargetWidth() const { return targetWidth_; }
    int getTargetHeight() const { return targetHeight_; }
    InterpolationMode getInterpolationMode() const { return mode_; }

private:
    int targetWidth_;
    int targetHeight_;
    InterpolationMode mode_;
};

} // namespace mini_image_pipe
