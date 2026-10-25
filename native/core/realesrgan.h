#pragma once

#include <functional>
#include <string>

#include "image_utils.h"
#include "model_manifest.h"
#include "ncnn/net.h"

namespace ic {

class RealEsrgan {
public:
    bool Load(const RealEsrganConfig& cfg, int threads, bool use_vulkan,
              std::string& error);

    int Scale() const { return cfg_.scale; }

    // Super-resolution to target_scale (1, 2 or cfg_.scale, typically 4).
    // The network always runs at cfg_.scale; tiles are downscaled before
    // accumulation so 2x mode never allocates a 4x float canvas.
    ImageRgb Upscale(const ImageRgb& image, int target_scale,
                     const std::function<void(float)>& percent_cb) const;

private:
    ncnn::Net net_;
    RealEsrganConfig cfg_;
    bool loaded_ = false;
};

}  // namespace ic
