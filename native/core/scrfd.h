#pragma once

#include <string>
#include <vector>

#include "image_utils.h"
#include "model_manifest.h"
#include "ncnn/net.h"

namespace ic {

struct Face {
    float x1, y1, x2, y2;
    float score;
    float kps[10];  // x0,y0,...x4,y4 in source-image coordinates
};

class Scrfd {
public:
    bool Load(const ScrfdConfig& cfg, int threads, bool use_vulkan,
              std::string& error);

    // Letterboxed inference on a packed RGB image.
    std::vector<Face> Detect(const ImageRgb& image) const;

private:
    ncnn::Net net_;
    ScrfdConfig cfg_;
    bool loaded_ = false;
};

}  // namespace ic
