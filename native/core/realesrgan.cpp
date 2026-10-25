#include "realesrgan.h"

#include <algorithm>
#include <cmath>

namespace ic {
namespace {

uint8_t clamp_u8(float v) {
    return static_cast<uint8_t>(std::lround(std::min(255.f, std::max(0.f, v))));
}

}  // namespace

bool RealEsrgan::Load(const RealEsrganConfig& cfg, int threads, bool use_vulkan,
                      std::string& error) {
    cfg_ = cfg;
    net_.opt.use_vulkan_compute = use_vulkan;
    net_.opt.num_threads = threads;
    net_.opt.use_fp16_packed = true;
    net_.opt.use_fp16_storage = true;
    net_.opt.use_fp16_arithmetic = use_vulkan;

    if (net_.load_param(cfg.param_path.c_str()) != 0) {
        error = "RealESRGAN load_param failed: " + cfg.param_path;
        return false;
    }
    if (net_.load_model(cfg.bin_path.c_str()) != 0) {
        error = "RealESRGAN load_model failed: " + cfg.bin_path;
        return false;
    }
    loaded_ = true;
    return true;
}

ImageRgb RealEsrgan::Upscale(const ImageRgb& image, int target_scale,
                             const std::function<void(float)>& percent_cb) const {
    if (!loaded_) return ImageRgb();

    const int net_scale = cfg_.scale;
    if (target_scale != 1 && target_scale != 2 && target_scale != net_scale)
        return ImageRgb();
    const int scale = target_scale;
    const int tile = std::max(64, cfg_.tile);
    const int overlap = std::clamp(cfg_.tile_overlap, 0, tile / 4);

    // Pad to a whole number of tiles using edge replication.
    const int tiles_x = (image.w + tile - 1) / tile;
    const int tiles_y = (image.h + tile - 1) / tile;
    const int pad_w = tiles_x * tile;
    const int pad_h = tiles_y * tile;

    ncnn::Mat input(pad_w, pad_h, 3);
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < pad_h; ++y) {
            int sy = std::min(y, image.h - 1);
            float* row = input.channel(c).row<float>(y);
            for (int x = 0; x < pad_w; ++x) {
                int sx = std::min(x, image.w - 1);
                row[x] = image.at(sx, sy)[c] / 255.0f;
            }
        }
    }

    const int out_w = pad_w * scale;
    const int out_h = pad_h * scale;
    std::vector<float> accum(static_cast<size_t>(out_w) * out_h * 3, 0.f);
    std::vector<float> weight(static_cast<size_t>(out_w) * out_h, 0.f);

    // Feather ramp across an overlap strip at internal tile edges.
    auto ramp = [&](int i, int length, bool low_side, bool high_side) -> float {
        float v = 1.f;
        if (low_side && i < overlap) v *= static_cast<float>(i + 1) / (overlap + 1);
        int from_end = length - 1 - i;
        if (high_side && from_end < overlap)
            v *= static_cast<float>(from_end + 1) / (overlap + 1);
        return v;
    };

    int total = tiles_x * tiles_y;
    int done = 0;
    for (int ty = 0; ty < tiles_y; ++ty) {
        for (int tx = 0; tx < tiles_x; ++tx) {
            ncnn::Mat tile_in(tile, tile, 3);
            for (int c = 0; c < 3; ++c) {
                for (int y = 0; y < tile; ++y) {
                    const float* src = input.channel(c).row<const float>(ty * tile + y)
                                       + tx * tile;
                    std::copy(src, src + tile, tile_in.channel(c).row<float>(y));
                }
            }

            ncnn::Extractor ex = net_.create_extractor();
            ex.set_light_mode(true);
            ex.input(cfg_.input_blob.c_str(), tile_in);
            ncnn::Mat tile_out;
            if (ex.extract(cfg_.output_blob.c_str(), tile_out) != 0) return ImageRgb();

            const int tw = tile * scale;
            const int th = tile * scale;

            // Downscale the tile locally when the user asked for < net_scale.
            std::vector<float> tile_resized;
            const float* out_r = nullptr;
            const float* out_g = nullptr;
            const float* out_b = nullptr;
            if (scale == net_scale) {
                out_r = tile_out.channel(0);
                out_g = tile_out.channel(1);
                out_b = tile_out.channel(2);
            } else {
                const int nw = tile * net_scale;
                tile_resized.resize(static_cast<size_t>(tw) * th * 3);
                for (int c = 0; c < 3; ++c) {
                    const float* src_c = tile_out.channel(c);
                    for (int y = 0; y < th; ++y) {
                        double sy = (y + 0.5) * net_scale / scale - 0.5;
                        int y0 = static_cast<int>(std::floor(sy));
                        int y1 = std::min(y0 + 1, nw - 1);
                        y0 = std::clamp(y0, 0, nw - 1);
                        float fy = static_cast<float>(sy - y0);
                        for (int x = 0; x < tw; ++x) {
                            double sx = (x + 0.5) * net_scale / scale - 0.5;
                            int x0 = static_cast<int>(std::floor(sx));
                            int x1 = std::min(x0 + 1, nw - 1);
                            x0 = std::clamp(x0, 0, nw - 1);
                            float fx = static_cast<float>(sx - x0);
                            float v = (1 - fx) * (1 - fy) * src_c[y0 * nw + x0]
                                    + fx * (1 - fy) * src_c[y0 * nw + x1]
                                    + (1 - fx) * fy * src_c[y1 * nw + x0]
                                    + fx * fy * src_c[y1 * nw + x1];
                            tile_resized[(static_cast<size_t>(y) * tw + x) * 3 + c] = v;
                        }
                    }
                }
            }

            for (int y = 0; y < th; ++y) {
                float wy = ramp(y / scale, tile, ty > 0, ty < tiles_y - 1);
                int oy = ty * tw + y;
                for (int x = 0; x < tw; ++x) {
                    float wx = ramp(x / scale, tile, tx > 0, tx < tiles_x - 1);
                    float wgt = wx * wy;
                    int ox = tx * tw + x;
                    size_t oi = static_cast<size_t>(oy) * out_w + ox;
                    size_t ti = static_cast<size_t>(y) * tw + x;
                    if (scale == net_scale) {
                        accum[oi * 3 + 0] += out_r[ti] * wgt;
                        accum[oi * 3 + 1] += out_g[ti] * wgt;
                        accum[oi * 3 + 2] += out_b[ti] * wgt;
                    } else {
                        accum[oi * 3 + 0] += tile_resized[ti * 3 + 0] * wgt;
                        accum[oi * 3 + 1] += tile_resized[ti * 3 + 1] * wgt;
                        accum[oi * 3 + 2] += tile_resized[ti * 3 + 2] * wgt;
                    }
                    weight[oi] += wgt;
                }
            }

            ++done;
            if (percent_cb) percent_cb(static_cast<float>(done) / total);
        }
    }

    ImageRgb result(image.w * scale, image.h * scale);
    for (int y = 0; y < result.h; ++y) {
        for (int x = 0; x < result.w; ++x) {
            size_t i = static_cast<size_t>(y) * out_w + x;
            float inv = weight[i] > 1e-6f ? 1.f / weight[i] : 0.f;
            uint8_t* d = result.at(x, y);
            d[0] = clamp_u8(accum[i * 3 + 0] * inv * 255.f);
            d[1] = clamp_u8(accum[i * 3 + 1] * inv * 255.f);
            d[2] = clamp_u8(accum[i * 3 + 2] * inv * 255.f);
        }
    }
    return result;
}

}  // namespace ic
