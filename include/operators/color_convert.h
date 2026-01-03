#pragma once

#include "operator.h"
#include "types.h"

namespace mini_image_pipe {

class ColorConvertOperator : public IOperator {
public:
    explicit ColorConvertOperator(ColorConversionType type);
    ~ColorConvertOperator() override = default;

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

    const char* getName() const override { return "ColorConvert"; }

    ColorConversionType getConversionType() const { return type_; }

    // Luminance weights for RGB to Grayscale
    static constexpr float kLumR = 0.299f;
    static constexpr float kLumG = 0.587f;
    static constexpr float kLumB = 0.114f;

private:
    ColorConversionType type_;
};

} // namespace mini_image_pipe
