#include "colorize.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace ic {

bool Colorizer::Load(const ColorizeConfig& cfg, int threads, std::string& error) {
    cfg_ = cfg;
    // CPU fp32: see the class comment. The graph is light (256x256 input) and
    // fp16 storage measurably shifts the predicted chroma.
    net_.opt.use_vulkan_compute = false;
    net_.opt.num_threads = threads;
    net_.opt.use_fp16_packed = false;
    net_.opt.use_fp16_storage = false;
    net_.opt.use_fp16_arithmetic = false;

    if (net_.load_param(cfg.param_path.c_str()) != 0) {
        error = "DDColor load_param failed: " + cfg.param_path;
        return false;
    }
    if (net_.load_model(cfg.bin_path.c_str()) != 0) {
        error = "DDColor load_model failed: " + cfg.bin_path;
        return false;
    }
    loaded_ = true;
    return true;
}

ImageRgb Colorizer::Colorize(const ImageRgb& image) const {
    const int size = cfg_.size;
    ImageRgb out(image.w, image.h);
    if (!loaded_ || image.w <= 0 || image.h <= 0) return image;

    // Original L channel at full resolution, and the L of a resized copy that
    // the network actually sees.
    const size_t n = static_cast<size_t>(image.w) * image.h;
    std::vector<float> l_full(n);
    for (int y = 0; y < image.h; ++y) {
        for (int x = 0; x < image.w; ++x) {
            const uint8_t* px = image.at(x, y);
            float L, A, B;
            rgb_to_lab(px[0], px[1], px[2], L, A, B);
            l_full[static_cast<size_t>(y) * image.w + x] = L;
        }
    }

    // Neutral grey RGB from the resized L (Lab with a = b = 0), in [0,1].
    ncnn::Mat input(size, size, 3);
    for (int y = 0; y < size; ++y) {
        const double sy = (y + 0.5) * image.h / size - 0.5;
        const int y0 = std::min(std::max(static_cast<int>(std::floor(sy)), 0), image.h - 1);
        const int y1 = std::min(y0 + 1, image.h - 1);
        const float fy = static_cast<float>(sy - std::floor(sy));
        for (int x = 0; x < size; ++x) {
            const double sx = (x + 0.5) * image.w / size - 0.5;
            const int x0 = std::min(std::max(static_cast<int>(std::floor(sx)), 0), image.w - 1);
            const int x1 = std::min(x0 + 1, image.w - 1);
            const float fx = static_cast<float>(sx - std::floor(sx));

            const float l00 = l_full[static_cast<size_t>(y0) * image.w + x0];
            const float l10 = l_full[static_cast<size_t>(y0) * image.w + x1];
            const float l01 = l_full[static_cast<size_t>(y1) * image.w + x0];
            const float l11 = l_full[static_cast<size_t>(y1) * image.w + x1];
            const float L = (1 - fx) * (1 - fy) * l00 + fx * (1 - fy) * l10 +
                            (1 - fx) * fy * l01 + fx * fy * l11;

            float r, g, b;
            lab_to_rgb(L, 0.f, 0.f, r, g, b);
            input.channel(0).row(y)[x] = r / 255.0f;
            input.channel(1).row(y)[x] = g / 255.0f;
            input.channel(2).row(y)[x] = b / 255.0f;
        }
    }

    ncnn::Mat ab;
    {
        ncnn::Extractor ex = net_.create_extractor();
        ex.set_light_mode(true);
        ex.input(cfg_.input_blob.c_str(), input);
        if (ex.extract(cfg_.output_blob.c_str(), ab) != 0 || ab.w != size ||
            ab.h != size || ab.c < 2) {
            return image;
        }
    }

    // Upsample ab to the original resolution and recombine with the original L.
    for (int y = 0; y < image.h; ++y) {
        const double sy = (y + 0.5) * size / image.h - 0.5;
        const int y0 = std::min(std::max(static_cast<int>(std::floor(sy)), 0), size - 1);
        const int y1 = std::min(y0 + 1, size - 1);
        const float fy = static_cast<float>(sy - std::floor(sy));
        for (int x = 0; x < image.w; ++x) {
            const double sx = (x + 0.5) * size / image.w - 0.5;
            const int x0 = std::min(std::max(static_cast<int>(std::floor(sx)), 0), size - 1);
            const int x1 = std::min(x0 + 1, size - 1);
            const float fx = static_cast<float>(sx - std::floor(sx));

            auto sample = [&](int c) {
                const float* ch = ab.channel(c);
                const float v00 = ch[y0 * size + x0];
                const float v10 = ch[y0 * size + x1];
                const float v01 = ch[y1 * size + x0];
                const float v11 = ch[y1 * size + x1];
                return (1 - fx) * (1 - fy) * v00 + fx * (1 - fy) * v10 +
                       (1 - fx) * fy * v01 + fx * fy * v11;
            };
            const float A = sample(0);
            const float B = sample(1);
            const float L = l_full[static_cast<size_t>(y) * image.w + x];

            float r, g, b;
            lab_to_rgb(L, A, B, r, g, b);
            uint8_t* d = out.at(x, y);
            d[0] = static_cast<uint8_t>(std::lround(std::min(255.f, std::max(0.f, r))));
            d[1] = static_cast<uint8_t>(std::lround(std::min(255.f, std::max(0.f, g))));
            d[2] = static_cast<uint8_t>(std::lround(std::min(255.f, std::max(0.f, b))));
        }
    }
    return out;
}

}  // namespace ic
