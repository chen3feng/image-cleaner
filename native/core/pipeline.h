#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "codeformer.h"
#include "colorize.h"
#include "image_utils.h"
#include "inpaint.h"
#include "model_manifest.h"
#include "realesrgan.h"
#include "scrfd.h"

namespace ic {

struct Progress {
    int stage = 0;       // IC_STAGE_*
    float percent = 0.f; // 0..1
    std::string message;
};

using ProgressFn = std::function<void(const Progress&)>;

class Pipeline {
public:
    bool Init(const std::string& model_dir, const std::string& manifest_json,
              int threads, bool use_vulkan, std::string& error);

    // Produces an RGBA frame at src * (mode == FACES ? 1 : target_scale).
    // dst must have capacity out_w*out_h*4 bytes; returns false on error.
    bool Enhance(const uint8_t* src_rgba, int src_w, int src_h, int src_stride,
                 uint8_t* dst_rgba, int dst_capacity,
                 int mode, float fidelity, int target_scale,
                 int& out_w, int& out_h, std::string& error,
                 const ProgressFn& progress = {});

    // Regenerates the painted area (mask >= 128) with LaMa; same output size.
    bool Inpaint(const uint8_t* src_rgba, int src_w, int src_h, int src_stride,
                 const uint8_t* mask, int mask_stride,
                 uint8_t* dst_rgba, int dst_capacity,
                 std::string& error, const ProgressFn& progress = {});

private:
    ModelManifest manifest_;
    int threads_ = 4;
    bool use_vulkan_ = false;
    std::unique_ptr<Scrfd> scrfd_;
    std::unique_ptr<CodeFormer> codeformer_;
    std::unique_ptr<RealEsrgan> esrgan_;
    std::unique_ptr<Colorizer> colorizer_;
    std::unique_ptr<Inpainter> inpainter_;

    bool ensure_scrfd(std::string& error);
    bool ensure_codeformer(std::string& error);
    bool ensure_esrgan(std::string& error);
    bool ensure_colorizer(std::string& error);
    bool ensure_inpainter(std::string& error);

    // Paste a restored 512x512 face into the canvas at output scale s.
    void paste_face(ImageRgb& canvas, int scale, const Face& face,
                    const ImageRgb& restored, const std::vector<float>& mask512);
};

}  // namespace ic
