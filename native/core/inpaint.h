#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "image_utils.h"
#include "model_manifest.h"
#include "ncnn/net.h"

namespace ic {

// LaMa (Samsung Research) large-mask inpainting.
//
// The exported graph is fixed at 512x512 and takes 4 channels: RGB with the
// masked area zeroed out, plus a binary mask. Rather than squashing the whole
// photo into 512 (which would blur the fill), the caller crops a square
// window around the painted region, runs the 512 model on it and composites
// the result back, so detail outside the window is untouched.
class Inpainter {
public:
    bool Load(const InpaintConfig& cfg, int threads, bool use_vulkan,
              std::string& error);

    // mask: one byte per pixel, 0..255 (>=128 is a hole), same dims as image.
    // Returns the image with the masked area regenerated.
    ImageRgb Inpaint(const ImageRgb& image, const uint8_t* mask, int mask_stride,
                     const std::function<void(float)>& percent_cb) const;

private:
    ncnn::Net net_;
    InpaintConfig cfg_;
    bool loaded_ = false;
};

}  // namespace ic
