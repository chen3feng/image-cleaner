#include "image_utils.h"

#include <algorithm>
#include <cmath>

namespace ic {

void rgba_to_rgb(const uint8_t* rgba, int w, int h, int stride_bytes, ImageRgb& out) {
    out = ImageRgb(w, h);
    for (int y = 0; y < h; ++y) {
        const uint8_t* src_row = rgba + static_cast<size_t>(y) * stride_bytes;
        uint8_t* dst_row = out.at(0, y);
        for (int x = 0; x < w; ++x) {
            dst_row[x * 3 + 0] = src_row[x * 4 + 0];  // R
            dst_row[x * 3 + 1] = src_row[x * 4 + 1];  // G
            dst_row[x * 3 + 2] = src_row[x * 4 + 2];  // B
        }
    }
}

ImageRgb rgb_copy(const ImageRgb& src) {
    return src;
}

void rgb_to_rgba(const ImageRgb& img, uint8_t* rgba, int stride_bytes) {
    for (int y = 0; y < img.h; ++y) {
        uint8_t* row = rgba + static_cast<size_t>(y) * stride_bytes;
        const uint8_t* s = img.at(0, y);
        for (int x = 0; x < img.w; ++x) {
            row[x * 4 + 0] = s[x * 3 + 0];
            row[x * 4 + 1] = s[x * 3 + 1];
            row[x * 4 + 2] = s[x * 3 + 2];
            row[x * 4 + 3] = 255;
        }
    }
}

bool similarity_5pt(const float src_xy[10], const float dst_xy[10],
                    Affine2& src_to_dst) {
    double smx = 0, smy = 0, dmx = 0, dmy = 0;
    for (int i = 0; i < 5; ++i) {
        smx += src_xy[i * 2 + 0];
        smy += src_xy[i * 2 + 1];
        dmx += dst_xy[i * 2 + 0];
        dmy += dst_xy[i * 2 + 1];
    }
    smx /= 5.0;
    smy /= 5.0;
    dmx /= 5.0;
    dmy /= 5.0;

    double num_a = 0, num_b = 0, den = 0;
    for (int i = 0; i < 5; ++i) {
        double sx = src_xy[i * 2 + 0] - smx;
        double sy = src_xy[i * 2 + 1] - smy;
        double dx = dst_xy[i * 2 + 0] - dmx;
        double dy = dst_xy[i * 2 + 1] - dmy;
        num_a += sx * dx + sy * dy;
        // Model: [dx dy] = [a -b; b a][sx sy] + t
        num_b += sx * dy - sy * dx;
        den += sx * sx + sy * sy;
    }
    if (den < 1e-9) return false;
    double a = num_a / den;
    double b = num_b / den;
    double tx = dmx - a * smx + b * smy;
    double ty = dmy - b * smx - a * smy;

    src_to_dst.m[0] = a;
    src_to_dst.m[1] = -b;
    src_to_dst.m[2] = tx;
    src_to_dst.m[3] = b;
    src_to_dst.m[4] = a;
    src_to_dst.m[5] = ty;
    return std::isfinite(a) && std::isfinite(b) && std::isfinite(tx) && std::isfinite(ty);
}

bool affine_invert(const Affine2& in, Affine2& out) {
    double a = in.m[0], b = in.m[1], tx = in.m[2];
    double c = in.m[3], d = in.m[4], ty = in.m[5];
    double det = a * d - b * c;
    if (std::fabs(det) < 1e-12) return false;
    double inv = 1.0 / det;
    out.m[0] = d * inv;
    out.m[1] = -b * inv;
    out.m[2] = (b * ty - d * tx) * inv;
    out.m[3] = -c * inv;
    out.m[4] = a * inv;
    out.m[5] = (c * tx - a * ty) * inv;
    return true;
}

namespace {

inline bool sample_bilinear(const uint8_t* src, int sw, int sh,
                            double sx, double sy, int channel, uint8_t out[3]) {
    if (sx < 0 || sy < 0 || sx > sw - 1 || sy > sh - 1) return false;
    int x0 = static_cast<int>(std::floor(sx));
    int y0 = static_cast<int>(std::floor(sy));
    int x1 = std::min(x0 + 1, sw - 1);
    int y1 = std::min(y0 + 1, sh - 1);
    double fx = sx - x0;
    double fy = sy - y0;
    double w00 = (1 - fx) * (1 - fy);
    double w10 = fx * (1 - fy);
    double w01 = (1 - fx) * fy;
    double w11 = fx * fy;
    for (int c = 0; c < channel; ++c) {
        double v = w00 * src[(static_cast<size_t>(y0) * sw + x0) * channel + c]
                 + w10 * src[(static_cast<size_t>(y0) * sw + x1) * channel + c]
                 + w01 * src[(static_cast<size_t>(y1) * sw + x0) * channel + c]
                 + w11 * src[(static_cast<size_t>(y1) * sw + x1) * channel + c];
        out[c] = static_cast<uint8_t>(std::lround(std::min(255.0, std::max(0.0, v))));
    }
    return true;
}

inline float sample_bilinear_f32(const float* src, int sw, int sh,
                                 double sx, double sy, bool* ok) {
    if (sx < 0 || sy < 0 || sx > sw - 1 || sy > sh - 1) {
        if (ok) *ok = false;
        return 0.f;
    }
    int x0 = static_cast<int>(std::floor(sx));
    int y0 = static_cast<int>(std::floor(sy));
    int x1 = std::min(x0 + 1, sw - 1);
    int y1 = std::min(y0 + 1, sh - 1);
    float fx = static_cast<float>(sx - x0);
    float fy = static_cast<float>(sy - y0);
    float v = (1 - fx) * (1 - fy) * src[y0 * sw + x0]
            + fx * (1 - fy) * src[y0 * sw + x1]
            + (1 - fx) * fy * src[y1 * sw + x0]
            + fx * fy * src[y1 * sw + x1];
    if (ok) *ok = true;
    return v;
}

}  // namespace

void warp_rgb(const ImageRgb& src, ImageRgb& dst, const Affine2& dst_to_src) {
    const double* m = dst_to_src.m;
    for (int y = 0; y < dst.h; ++y) {
        for (int x = 0; x < dst.w; ++x) {
            double sx = m[0] * x + m[1] * y + m[2];
            double sy = m[3] * x + m[4] * y + m[5];
            uint8_t sample[3];
            if (sample_bilinear(src.px.data(), src.w, src.h, sx, sy, 3, sample)) {
                uint8_t* d = dst.at(x, y);
                d[0] = sample[0];
                d[1] = sample[1];
                d[2] = sample[2];
            }
        }
    }
}

void warp_rgb_clamped(const ImageRgb& src, ImageRgb& dst, const Affine2& dst_to_src) {
    const double* m = dst_to_src.m;
    const double max_x = src.w - 1.0, max_y = src.h - 1.0;
    for (int y = 0; y < dst.h; ++y) {
        for (int x = 0; x < dst.w; ++x) {
            double sx = m[0] * x + m[1] * y + m[2];
            double sy = m[3] * x + m[4] * y + m[5];
            sx = std::min(std::max(sx, 0.0), max_x);
            sy = std::min(std::max(sy, 0.0), max_y);
            uint8_t sample[3];
            if (sample_bilinear(src.px.data(), src.w, src.h, sx, sy, 3, sample)) {
                uint8_t* d = dst.at(x, y);
                d[0] = sample[0];
                d[1] = sample[1];
                d[2] = sample[2];
            }
        }
    }
}

void warp_mask_f32(const std::vector<float>& src, int sw, int sh,
                   std::vector<float>& dst, int dw, int dh, const Affine2& dst_to_src) {
    dst.assign(static_cast<size_t>(dw) * dh, 0.f);
    const double* m = dst_to_src.m;
    for (int y = 0; y < dh; ++y) {
        for (int x = 0; x < dw; ++x) {
            double sx = m[0] * x + m[1] * y + m[2];
            double sy = m[3] * x + m[4] * y + m[5];
            bool ok = false;
            float v = sample_bilinear_f32(src.data(), sw, sh, sx, sy, &ok);
            if (ok) dst[static_cast<size_t>(y) * dw + x] = v;
        }
    }
}

std::vector<float> warp_validity(int sw, int sh, int dw, int dh,
                                 const Affine2& dst_to_src, float feather) {
    std::vector<float> valid(static_cast<size_t>(dw) * dh, 0.f);
    const double* m = dst_to_src.m;
    const float band = std::max(1.f, feather);
    for (int y = 0; y < dh; ++y) {
        for (int x = 0; x < dw; ++x) {
            const double sx = m[0] * x + m[1] * y + m[2];
            const double sy = m[3] * x + m[4] * y + m[5];
            // Signed margin to the nearest frame border, ramped over `band`
            // so the blend fades out before reaching clamped edge pixels.
            const double inside = std::min({sx, sy, (sw - 1) - sx, (sh - 1) - sy});
            valid[static_cast<size_t>(y) * dw + x] =
                static_cast<float>(std::min(1.0, std::max(0.0, inside / band)));
        }
    }
    return valid;
}

ImageRgb resize_bilinear(const ImageRgb& src, int new_w, int new_h) {
    if (new_w == src.w && new_h == src.h) return src;
    ImageRgb dst(new_w, new_h);
    const double sx_ratio = static_cast<double>(src.w) / new_w;
    const double sy_ratio = static_cast<double>(src.h) / new_h;
    for (int y = 0; y < new_h; ++y) {
        double sy = (y + 0.5) * sy_ratio - 0.5;
        for (int x = 0; x < new_w; ++x) {
            double sx = (x + 0.5) * sx_ratio - 0.5;
            uint8_t sample[3];
            if (!sample_bilinear(src.px.data(), src.w, src.h, sx, sy, 3, sample)) {
                // Clamp edge sampling for half-pixel centre convention
                sx = std::min(std::max(sx, 0.0), src.w - 1.0);
                sy = std::min(std::max(sy, 0.0), src.h - 1.0);
                sample_bilinear(src.px.data(), src.w, src.h, sx, sy, 3, sample);
            }
            uint8_t* d = dst.at(x, y);
            d[0] = sample[0];
            d[1] = sample[1];
            d[2] = sample[2];
        }
    }
    return dst;
}

std::vector<float> ellipse_feather_mask(int size, float cx, float cy,
                                        float rx, float ry, float feather) {
    std::vector<float> mask(static_cast<size_t>(size) * size, 0.f);
    const float band = std::max(1.f, feather);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            float qx = (x - cx) / rx;
            float qy = (y - cy) / ry;
            float q = qx * qx + qy * qy;
            float a = (1.0f - q) * (rx / band);
            mask[static_cast<size_t>(y) * size + x] =
                std::min(1.f, std::max(0.f, a));
        }
    }
    return mask;
}

void arcface_template_5pt(float out_xy[10], int size) {
    // ArcFace reference landmarks for 112x112
    static const float kTemplate112[10] = {
        38.2946f, 51.6963f,
        73.5318f, 51.6963f,
        56.0252f, 71.7366f,
        41.5493f, 92.3655f,
        70.7299f, 92.3655f};
    float scale = static_cast<float>(size) / 112.0f;
    for (int i = 0; i < 10; ++i) out_xy[i] = kTemplate112[i] * scale;
}

}  // namespace ic
