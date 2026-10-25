#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ic {

// Packed RGB, 3 bytes per pixel.
struct ImageRgb {
    int w = 0;
    int h = 0;
    std::vector<uint8_t> px;  // size w*h*3

    ImageRgb() = default;
    ImageRgb(int width, int height) : w(width), h(height), px(width * height * 3, 0) {}

    uint8_t* at(int x, int y) { return px.data() + (static_cast<size_t>(y) * w + x) * 3; }
    const uint8_t* at(int x, int y) const {
        return px.data() + (static_cast<size_t>(y) * w + x) * 3;
    }
};

// 2x3 affine, row major: maps point (x,y) to (m0*x+m1*y+m2, m3*x+m4*y+m5)
struct Affine2 {
    double m[6] = {1, 0, 0, 0, 1, 0};
};

void rgba_to_rgb(const uint8_t* rgba, int w, int h, int stride_bytes, ImageRgb& out);
ImageRgb rgb_copy(const ImageRgb& src);
void rgb_to_rgba(const ImageRgb& img, uint8_t* rgba, int stride_bytes);

// Least-squares similarity (rotation+uniform scale+translation) from 5 point
// pairs. Returns false if the configuration is degenerate.
bool similarity_5pt(const float src_xy[10], const float dst_xy[10], Affine2& src_to_dst);

// Inverts a non-singular 2x3 affine.
bool affine_invert(const Affine2& in, Affine2& out);

// Bilinear warp. dst_to_src maps destination pixel coordinates to source
// coordinates; pixels sampling outside src are left untouched (caller zeroes).
void warp_rgb(const ImageRgb& src, ImageRgb& dst, const Affine2& dst_to_src);

// Same warp, but out-of-frame samples clamp to the nearest edge pixel instead
// of leaving black. Face alignment of subjects near the photo border would
// otherwise feed hard black wedges to CodeFormer, which turns them into
// noisy/bright garbage that the paste-back mask then blends in.
void warp_rgb_clamped(const ImageRgb& src, ImageRgb& dst, const Affine2& dst_to_src);
void warp_mask_f32(const std::vector<float>& src, int sw, int sh,
                   std::vector<float>& dst, int dw, int dh, const Affine2& dst_to_src);

// Soft validity map in destination space: 1 where the inverse-warped sample
// falls inside the source frame, ramping to 0 over `feather` pixels outside.
// Multiplying a paste mask by this keeps extrapolated (edge-clamped) areas --
// which CodeFormer reconstructs as garbage -- from being blended in.
std::vector<float> warp_validity(int sw, int sh, int dw, int dh,
                                 const Affine2& dst_to_src, float feather);

ImageRgb resize_bilinear(const ImageRgb& src, int new_w, int new_h);

// Feathered ellipse mask, values in [0,1].
std::vector<float> ellipse_feather_mask(int size, float cx, float cy,
                                        float rx, float ry, float feather);

// ArcFace 5-point reference template scaled from 112x112 to size x size.
void arcface_template_5pt(float out_xy[10], int size);

}  // namespace ic
