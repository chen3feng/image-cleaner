#pragma once

#include <functional>
#include <string>
#include <vector>

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
// The network input is only 256x256, which is very coarse for a phone photo.
// For images larger than 640 px we therefore run it on a 448 px working copy
// split into up to 2x2 overlapping 256 px tiles, and use a whole-image pass as
// the colour anchor for each tile (otherwise neighbouring tiles pick different
// colours for the same object and the seams are obvious). This costs 5 passes
// instead of 1 but keeps the colour detail at roughly twice the resolution.
//
// Runs on CPU in fp32: fp16 storage shifts the chroma noticeably (mean ab
// error 2.2 vs 0.13), and the Vulkan path without fp16 is slower than CPU.
class Colorizer {
public:
    bool Load(const ColorizeConfig& cfg, int threads, std::string& error);

    // Returns a colorized copy of `image` at the same resolution.
    // `strength` multiplies the predicted chroma (1 = model output; values
    // above 1 give the often-muted DDColor result more vivid colours).
    ImageRgb Colorize(const ImageRgb& image, float strength = 1.f,
                      const std::function<void(float)>& progress = {}) const;

private:
    // l (w x h) -> ab (2, size, size).
    bool PredictAb(const std::vector<float>& l, int w, int h, ncnn::Mat& ab) const;

    ncnn::Net net_;
    ColorizeConfig cfg_;
    bool loaded_ = false;
};

}  // namespace ic
