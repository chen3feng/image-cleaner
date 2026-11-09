#include "colorize.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace ic {
namespace {

// Above this long edge a single 256x256 pass throws away too much detail.
constexpr int kTileThreshold = 640;
// Working resolution for the tiled pass: 2x256 with 64 px of overlap.
constexpr int kWorkLong = 448;

inline float sample_plane(const std::vector<float>& src, int sw, int sh,
                          float sx, float sy) {
    sx = std::min(std::max(sx, 0.f), static_cast<float>(sw - 1));
    sy = std::min(std::max(sy, 0.f), static_cast<float>(sh - 1));
    const int x0 = static_cast<int>(std::floor(sx));
    const int y0 = static_cast<int>(std::floor(sy));
    const int x1 = std::min(x0 + 1, sw - 1);
    const int y1 = std::min(y0 + 1, sh - 1);
    const float fx = sx - x0, fy = sy - y0;
    const float v00 = src[static_cast<size_t>(y0) * sw + x0];
    const float v10 = src[static_cast<size_t>(y0) * sw + x1];
    const float v01 = src[static_cast<size_t>(y1) * sw + x0];
    const float v11 = src[static_cast<size_t>(y1) * sw + x1];
    return (1 - fx) * (1 - fy) * v00 + fx * (1 - fy) * v10 +
           (1 - fx) * fy * v01 + fx * fy * v11;
}

inline void sample_ab(const ncnn::Mat& ab, float sx, float sy,
                      float& a, float& b) {
    const int sw = ab.w, sh = ab.h;
    sx = std::min(std::max(sx, 0.f), static_cast<float>(sw - 1));
    sy = std::min(std::max(sy, 0.f), static_cast<float>(sh - 1));
    const int x0 = static_cast<int>(std::floor(sx));
    const int y0 = static_cast<int>(std::floor(sy));
    const int x1 = std::min(x0 + 1, sw - 1);
    const int y1 = std::min(y0 + 1, sh - 1);
    const float fx = sx - x0, fy = sy - y0;
    const float* ca = ab.channel(0);
    const float* cb = ab.channel(1);
    auto s = [&](const float* c) {
        return (1 - fx) * (1 - fy) * c[y0 * sw + x0] + fx * (1 - fy) * c[y0 * sw + x1] +
               (1 - fx) * fy * c[y1 * sw + x0] + fx * fy * c[y1 * sw + x1];
    };
    a = s(ca);
    b = s(cb);
}

// Raised-cosine blend window with a small floor: the floor keeps edge pixels
// recoverable when only one tile covers them.
inline float axis_weight(int i, int n) {
    const float t = (i + 0.5f) / static_cast<float>(n);
    return 0.02f + 0.98f * std::sin(3.14159265f * t);
}

}  // namespace

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

bool Colorizer::PredictAb(const std::vector<float>& l, int w, int h,
                          ncnn::Mat& ab) const {
    const int size = cfg_.size;
    ncnn::Mat input(size, size, 3);
    for (int y = 0; y < size; ++y) {
        const float sy = (y + 0.5f) * h / size - 0.5f;
        for (int x = 0; x < size; ++x) {
            const float sx = (x + 0.5f) * w / size - 0.5f;
            const float L = sample_plane(l, w, h, sx, sy);
            float r, g, b;
            lab_to_rgb(L, 0.f, 0.f, r, g, b);
            input.channel(0).row(y)[x] = r / 255.0f;
            input.channel(1).row(y)[x] = g / 255.0f;
            input.channel(2).row(y)[x] = b / 255.0f;
        }
    }

    ncnn::Extractor ex = net_.create_extractor();
    ex.set_light_mode(true);
    ex.input(cfg_.input_blob.c_str(), input);
    return ex.extract(cfg_.output_blob.c_str(), ab) == 0 && ab.w == size &&
           ab.h == size && ab.c >= 2;
}

ImageRgb Colorizer::Colorize(const ImageRgb& image, float strength,
                             const std::function<void(float)>& progress) const {
    ImageRgb out(image.w, image.h);
    if (!loaded_ || image.w <= 0 || image.h <= 0) return image;

    const int size = cfg_.size;
    const int iw = image.w, ih = image.h;

    // Original L channel at full resolution.
    std::vector<float> l_full(static_cast<size_t>(iw) * ih);
    for (int y = 0; y < ih; ++y) {
        for (int x = 0; x < iw; ++x) {
            const uint8_t* px = image.at(x, y);
            float L, A, B;
            rgb_to_lab(px[0], px[1], px[2], L, A, B);
            l_full[static_cast<size_t>(y) * iw + x] = L;
        }
    }

    strength = std::min(2.5f, std::max(0.2f, strength));

    auto finish = [&](float a, float b, float L, int x, int y) {
        float r, g, bl;
        lab_to_rgb(L, a * strength, b * strength, r, g, bl);
        uint8_t* d = out.at(x, y);
        d[0] = static_cast<uint8_t>(std::lround(std::min(255.f, std::max(0.f, r))));
        d[1] = static_cast<uint8_t>(std::lround(std::min(255.f, std::max(0.f, g))));
        d[2] = static_cast<uint8_t>(std::lround(std::min(255.f, std::max(0.f, bl))));
    };

    // Small images: one whole-image pass is enough.
    if (std::max(iw, ih) <= kTileThreshold) {
        ncnn::Mat ab;
        if (!PredictAb(l_full, iw, ih, ab)) return image;
        for (int y = 0; y < ih; ++y) {
            const float sy = (y + 0.5f) * size / ih - 0.5f;
            for (int x = 0; x < iw; ++x) {
                const float sx = (x + 0.5f) * size / iw - 0.5f;
                float a, b;
                sample_ab(ab, sx, sy, a, b);
                finish(a, b, l_full[static_cast<size_t>(y) * iw + x], x, y);
            }
        }
        if (progress) progress(1.f);
        return out;
    }

    // Work at a reduced resolution so the tiles cover the image in 2x2.
    const int long_edge = std::max(iw, ih);
    const int work_w = std::max(1, static_cast<int>(
        std::lround(static_cast<double>(iw) * kWorkLong / long_edge)));
    const int work_h = std::max(1, static_cast<int>(
        std::lround(static_cast<double>(ih) * kWorkLong / long_edge)));
    std::vector<float> l_work(static_cast<size_t>(work_w) * work_h);
    for (int y = 0; y < work_h; ++y) {
        const float sy = (y + 0.5f) * ih / work_h - 0.5f;
        for (int x = 0; x < work_w; ++x) {
            const float sx = (x + 0.5f) * iw / work_w - 0.5f;
            l_work[static_cast<size_t>(y) * work_w + x] =
                sample_plane(l_full, iw, ih, sx, sy);
        }
    }

    // Whole-image anchor: tiles are shifted to its mean colour so the seams do
    // not show. It also serves as the fallback for any uncovered pixel.
    ncnn::Mat ab_global;
    if (!PredictAb(l_work, work_w, work_h, ab_global)) return image;
    std::vector<float> anchor(static_cast<size_t>(2) * work_w * work_h);
    for (int y = 0; y < work_h; ++y) {
        const float sy = (y + 0.5f) * size / work_h - 0.5f;
        for (int x = 0; x < work_w; ++x) {
            const float sx = (x + 0.5f) * size / work_w - 0.5f;
            float a, b;
            sample_ab(ab_global, sx, sy, a, b);
            const size_t i = static_cast<size_t>(y) * work_w + x;
            anchor[i] = a;
            anchor[static_cast<size_t>(work_w) * work_h + i] = b;
        }
    }

    std::vector<int> xs, ys;
    xs.push_back(0);
    if (work_w > size) xs.push_back(work_w - size);
    ys.push_back(0);
    if (work_h > size) ys.push_back(work_h - size);

    const int total = static_cast<int>(xs.size() * ys.size()) + 1;
    int done = 1;
    if (progress) progress(static_cast<float>(done) / total);

    std::vector<float> acc(static_cast<size_t>(2) * work_w * work_h, 0.f);
    std::vector<float> weight(static_cast<size_t>(work_w) * work_h, 0.f);
    const size_t plane = static_cast<size_t>(work_w) * work_h;

    for (int y0 : ys) {
        for (int x0 : xs) {
            const int x1 = std::min(x0 + size, work_w);
            const int y1 = std::min(y0 + size, work_h);
            const int tx = std::max(0, x1 - size);
            const int ty = std::max(0, y1 - size);
            const int tw = x1 - tx, th = y1 - ty;

            std::vector<float> crop(static_cast<size_t>(tw) * th);
            for (int y = 0; y < th; ++y) {
                for (int x = 0; x < tw; ++x) {
                    crop[static_cast<size_t>(y) * tw + x] =
                        l_work[static_cast<size_t>(ty + y) * work_w + tx + x];
                }
            }
            ncnn::Mat ab;
            if (!PredictAb(crop, tw, th, ab)) return image;

            // Tile ab at the work resolution, plus the mean shift towards the
            // global anchor over the same region.
            std::vector<float> tile(static_cast<size_t>(2) * tw * th);
            double sum_a = 0, sum_b = 0, ref_a = 0, ref_b = 0;
            for (int y = 0; y < th; ++y) {
                const float sy = (y + 0.5f) * size / th - 0.5f;
                for (int x = 0; x < tw; ++x) {
                    const float sx = (x + 0.5f) * size / tw - 0.5f;
                    float a, b;
                    sample_ab(ab, sx, sy, a, b);
                    const size_t i = static_cast<size_t>(y) * tw + x;
                    tile[i] = a;
                    tile[static_cast<size_t>(tw) * th + i] = b;
                    sum_a += a;
                    sum_b += b;
                    const size_t wi = static_cast<size_t>(ty + y) * work_w + tx + x;
                    ref_a += anchor[wi];
                    ref_b += anchor[plane + wi];
                }
            }
            const double n = static_cast<double>(tw) * th;
            const float shift_a = static_cast<float>(ref_a / n - sum_a / n);
            const float shift_b = static_cast<float>(ref_b / n - sum_b / n);

            for (int y = 0; y < th; ++y) {
                const float wy = axis_weight(y, th);
                for (int x = 0; x < tw; ++x) {
                    const float wgt = wy * axis_weight(x, tw);
                    const size_t i = static_cast<size_t>(y) * tw + x;
                    const size_t wi = static_cast<size_t>(ty + y) * work_w + tx + x;
                    acc[wi] += (tile[i] + shift_a) * wgt;
                    acc[plane + wi] += (tile[static_cast<size_t>(tw) * th + i] + shift_b) * wgt;
                    weight[wi] += wgt;
                }
            }
            ++done;
            if (progress) progress(static_cast<float>(done) / total);
        }
    }

    for (int y = 0; y < ih; ++y) {
        const float sy = (y + 0.5f) * work_h / ih - 0.5f;
        for (int x = 0; x < iw; ++x) {
            const float sx = (x + 0.5f) * work_w / iw - 0.5f;
            // Bilinear sample of the normalised accumulator.
            const float cx = std::min(std::max(sx, 0.f), static_cast<float>(work_w - 1));
            const float cy = std::min(std::max(sy, 0.f), static_cast<float>(work_h - 1));
            const int ax0 = static_cast<int>(std::floor(cx));
            const int ay0 = static_cast<int>(std::floor(cy));
            const int ax1 = std::min(ax0 + 1, work_w - 1);
            const int ay1 = std::min(ay0 + 1, work_h - 1);
            const float fx = cx - ax0, fy = cy - ay0;
            auto norm = [&](int xx, int yy, size_t base) {
                const size_t i = static_cast<size_t>(yy) * work_w + xx;
                const float wgt = std::max(weight[i], 1e-6f);
                return acc[base + i] / wgt;
            };
            const float a = (1 - fx) * (1 - fy) * norm(ax0, ay0, 0) +
                            fx * (1 - fy) * norm(ax1, ay0, 0) +
                            (1 - fx) * fy * norm(ax0, ay1, 0) +
                            fx * fy * norm(ax1, ay1, 0);
            const float b = (1 - fx) * (1 - fy) * norm(ax0, ay0, plane) +
                            fx * (1 - fy) * norm(ax1, ay0, plane) +
                            (1 - fx) * fy * norm(ax0, ay1, plane) +
                            fx * fy * norm(ax1, ay1, plane);
            finish(a, b, l_full[static_cast<size_t>(y) * iw + x], x, y);
        }
    }
    if (progress) progress(1.f);
    return out;
}

}  // namespace ic
