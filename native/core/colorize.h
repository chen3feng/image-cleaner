#pragma once

#include <string>

#include "image_utils.h"
#include "model_manifest.h"
#include "ncnn/net.h"

namespace ic {

// DDColor (ICCV 2023) photo colorization.
//
// The network takes a neutral-grey RGB image (built from the L channel of the
// input, resized to `size`) and predicts the two Lab chroma channels ab at the
// same resolution. The caller re-attaches those to the *original* L channel,
// exactly like the reference pipeline (ddcolor/pipeline.py).
//
// Runs on CPU in fp32: at 256x256 it is ~1 s on a phone, while the Vulkan
// path without fp16 storage is several times slower and fp16 storage shifts
// the chroma noticeably (mean ab error 2.2 vs 0.13).
class Colorizer {
public:
    bool Load(const ColorizeConfig& cfg, int threads, std::string& error);

    // Returns a colorized copy of `image` at the same resolution.
    ImageRgb Colorize(const ImageRgb& image) const;

private:
    ncnn::Net net_;
    ColorizeConfig cfg_;
    bool loaded_ = false;
};

}  // namespace ic
