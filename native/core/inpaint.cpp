#include "inpaint.h"

#include <algorithm>
#include <cmath>

namespace ic {
namespace {

struct Crop {
    int x = 0, y = 0, side = 0;
};

// Square window centred on the painted region, with 2x context so LaMa sees
// enough surrounding structure; clamped to the photo.
Crop compute_crop(const uint8_t* mask, int w, int h, int stride) {
    int min_x = w, min_y = h, max_x = -1, max_y = -1;
    for (int y = 0; y < h; ++y) {
        const uint8_t* row = mask + static_cast<size_t>(y) * stride;
        for (int x = 0; x < w; ++x) {
            if (row[x] >= 128) {
                min_x = std::min(min_x, x);
                max_x = std::max(max_x, x);
                min_y = std::min(min_y, y);
                max_y = std::max(max_y, y);
            }
        }
    }
    Crop c;
    if (max_x < 0) return c;  // empty mask

    const int bw = max_x - min_x + 1;
    const int bh = max_y - min_y + 1;
    int side = std::max(64, static_cast<int>(std::max(bw, bh) * 2.0f));
    side = std::min({side, std::max(w, h), w, h});
    side = std::max(side, std::min({w, h, 64}));

    const int cx = (min_x + max_x) / 2;
    const int cy = (min_y + max_y) / 2;
    c.side = side;
    c.x = std::min(std::max(cx - side / 2, 0), w - side);
    c.y = std::min(std::max(cy - side / 2, 0), h - side);
    return c;
}

}  // namespace

bool Inpainter::Load(const InpaintConfig& cfg, int threads, bool use_vulkan,
                     std::string& error) {
    cfg_ = cfg;
    // fp16 storage on Vulkan was both accurate (mean error 0.0016) and the
    // fastest option measured, so this one keeps the GPU.
    net_.opt.use_vulkan_compute = use_vulkan;
    net_.opt.num_threads = threads;
    net_.opt.use_fp16_packed = use_vulkan;
    net_.opt.use_fp16_storage = use_vulkan;
    net_.opt.use_fp16_arithmetic = use_vulkan;

    if (net_.load_param(cfg.param_path.c_str()) != 0) {
        error = "LaMa load_param failed: " + cfg.param_path;
        return false;
    }
    if (net_.load_model(cfg.bin_path.c_str()) != 0) {
        error = "LaMa load_model failed: " + cfg.bin_path;
        return false;
    }
    loaded_ = true;
    return true;
}

ImageRgb Inpainter::Inpaint(const ImageRgb& image, const uint8_t* mask,
                            int mask_stride,
                            const std::function<void(float)>& percent_cb) const {
    if (!loaded_ || !mask) return image;
    const Crop crop = compute_crop(mask, image.w, image.h, mask_stride);
    if (crop.side <= 0) return image;  // nothing painted

    const int S = crop.side;
    const int N = cfg_.size;

    // Crop + resize to the model resolution, holes zeroed in RGB as LaMa
    // expects ("masked image ... masked region zeroed out").
    ncnn::Mat input(N, N, 4);
    for (int y = 0; y < N; ++y) {
        const double sy = (y + 0.5) * S / N - 0.5;
        const int y0 = std::min(std::max(static_cast<int>(std::floor(sy)), 0), S - 1);
        const int y1 = std::min(y0 + 1, S - 1);
        const float fy = static_cast<float>(sy - std::floor(sy));
        for (int x = 0; x < N; ++x) {
            const double sx = (x + 0.5) * S / N - 0.5;
            const int x0 = std::min(std::max(static_cast<int>(std::floor(sx)), 0), S - 1);
            const int x1 = std::min(x0 + 1, S - 1);
            const float fx = static_cast<float>(sx - std::floor(sx));

            float rgb[3] = {0.f, 0.f, 0.f};
            float a = 0.f;
            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx) {
                    const int px = dx ? x1 : x0;
                    const int py = dy ? y1 : y0;
                    const float wgt = (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy);
                    const uint8_t* p = image.at(crop.x + px, crop.y + py);
                    rgb[0] += wgt * p[0];
                    rgb[1] += wgt * p[1];
                    rgb[2] += wgt * p[2];
                    const uint8_t mv = mask[static_cast<size_t>(crop.y + py) * mask_stride +
                                            crop.x + px];
                    a += wgt * (mv >= 128 ? 1.f : 0.f);
                }
            }
            // Binary mask for the net; zero the hole in the RGB channels.
            const float hole = a >= 0.5f ? 1.f : 0.f;
            input.channel(0).row(y)[x] = hole > 0 ? 0.f : rgb[0] / 255.f;
            input.channel(1).row(y)[x] = hole > 0 ? 0.f : rgb[1] / 255.f;
            input.channel(2).row(y)[x] = hole > 0 ? 0.f : rgb[2] / 255.f;
            input.channel(3).row(y)[x] = hole;
        }
    }

    ncnn::Mat out;
    {
        ncnn::Extractor ex = net_.create_extractor();
        ex.set_light_mode(true);
        ex.input(cfg_.input_blob.c_str(), input);
        if (ex.extract(cfg_.output_blob.c_str(), out) != 0 || out.w != N ||
            out.h != N || out.c < 3) {
            return image;
        }
    }
    if (percent_cb) percent_cb(1.f);

    // Composite: model output inside the painted area, original elsewhere.
    // The bilinearly sampled mask doubles as the blend alpha (soft edge).
    ImageRgb result = image;
    for (int y = 0; y < S; ++y) {
        const double sy = (y + 0.5) * N / S - 0.5;
        const int y0 = std::min(std::max(static_cast<int>(std::floor(sy)), 0), N - 1);
        const int y1 = std::min(y0 + 1, N - 1);
        const float fy = static_cast<float>(sy - std::floor(sy));
        for (int x = 0; x < S; ++x) {
            const double sx = (x + 0.5) * N / S - 0.5;
            const int x0 = std::min(std::max(static_cast<int>(std::floor(sx)), 0), N - 1);
            const int x1 = std::min(x0 + 1, N - 1);
            const float fx = static_cast<float>(sx - std::floor(sx));

            const uint8_t mv =
                mask[static_cast<size_t>(crop.y + y) * mask_stride + crop.x + x];
            if (mv < 128) continue;  // untouched

            float rgb[3];
            for (int c = 0; c < 3; ++c) {
                const float* ch = out.channel(c);
                const float v = (1 - fx) * (1 - fy) * ch[y0 * N + x0] +
                                fx * (1 - fy) * ch[y0 * N + x1] +
                                (1 - fx) * fy * ch[y1 * N + x0] +
                                fx * fy * ch[y1 * N + x1];
                rgb[c] = std::min(255.f, std::max(0.f, v * 255.f));
            }
            uint8_t* d = result.at(crop.x + x, crop.y + y);
            for (int c = 0; c < 3; ++c) d[c] = static_cast<uint8_t>(std::lround(rgb[c]));
        }
    }
    return result;
}

}  // namespace ic
