#pragma once

#include "operator.h"
#include "types.h"

namespace mini_image_pipe {

class GaussianBlurOperator : public IOperator {
public:
    explicit GaussianBlurOperator(GaussianKernelSize size, float sigma = 0.0f);
    ~GaussianBlurOperator() override;

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

    const char* getName() const override { return "GaussianBlur"; }

    // Setters for runtime configuration
    void setKernelSize(GaussianKernelSize size);
    void setSigma(float sigma);

    GaussianKernelSize getKernelSize() const { return kernelSize_; }
    float getSigma() const { return sigma_; }

private:
    GaussianKernelSize kernelSize_;
    float sigma_;
    float* d_kernel_ = nullptr;      // 1D kernel on device
    void* d_intermediate_ = nullptr; // Intermediate buffer for separable filter
    size_t intermediateSize_ = 0;

    void generateKernel();
    void freeResources();
};

} // namespace mini_image_pipe
