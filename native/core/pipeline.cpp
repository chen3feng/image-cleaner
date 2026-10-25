#include "pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>

#include <android/log.h>

#include "enhancer_c_api.h"

#define PIXELUP_LOGI(...) __android_log_print(ANDROID_LOG_INFO, "PixelUpNative", __VA_ARGS__)

namespace {
using Clock = std::chrono::steady_clock;
long long ms_since(Clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t).count();
}
}  // namespace

namespace ic {
namespace {

struct BilinearRgb {
    uint8_t v[3] = {0, 0, 0};
    bool inside = false;
};

inline BilinearRgb sample_rgb(const ImageRgb& img, double sx, double sy) {
    BilinearRgb r;
    if (sx < 0 || sy < 0 || sx > img.w - 1 || sy > img.h - 1) return r;
    int x0 = static_cast<int>(std::floor(sx));
    int y0 = static_cast<int>(std::floor(sy));
    int x1 = std::min(x0 + 1, img.w - 1);
    int y1 = std::min(y0 + 1, img.h - 1);
    double fx = sx - x0, fy = sy - y0;
    double w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy);
    double w01 = (1 - fx) * fy, w11 = fx * fy;
    for (int c = 0; c < 3; ++c) {
        double v = w00 * img.at(x0, y0)[c] + w10 * img.at(x1, y0)[c]
                 + w01 * img.at(x0, y1)[c] + w11 * img.at(x1, y1)[c];
        r.v[c] = static_cast<uint8_t>(std::lround(std::clamp(v, 0.0, 255.0)));
    }
    r.inside = true;
    return r;
}

inline double sample_mask(const std::vector<float>& m, int w, int h,
                          double sx, double sy) {
    if (sx < 0 || sy < 0 || sx > w - 1 || sy > h - 1) return 0.0;
    int x0 = static_cast<int>(std::floor(sx));
    int y0 = static_cast<int>(std::floor(sy));
    int x1 = std::min(x0 + 1, w - 1);
    int y1 = std::min(y0 + 1, h - 1);
    double fx = sx - x0, fy = sy - y0;
    double v = (1 - fx) * (1 - fy) * m[y0 * w + x0] + fx * (1 - fy) * m[y0 * w + x1]
             + (1 - fx) * fy * m[y1 * w + x0] + fx * fy * m[y1 * w + x1];
    return v;
}

}  // namespace

bool Pipeline::Init(const std::string& model_dir, const std::string& manifest_json,
                    int threads, bool use_vulkan, std::string& error) {
    threads_ = std::max(1, threads);
    use_vulkan_ = use_vulkan;
    const bool ok = ModelManifest::Load(model_dir, manifest_json, manifest_, error);
    PIXELUP_LOGI("init: threads=%d request_vulkan=%d gpu_count=%d", threads_,
                 use_vulkan ? 1 : 0, ncnn::get_gpu_count());
    return ok;
}

bool Pipeline::ensure_scrfd(std::string& error) {
    if (!scrfd_) {
        scrfd_ = std::make_unique<Scrfd>();
        if (!scrfd_->Load(manifest_.scrfd, threads_, use_vulkan_, error)) return false;
    }
    return true;
}

bool Pipeline::ensure_codeformer(std::string& error) {
    if (!codeformer_) {
        codeformer_ = std::make_unique<CodeFormer>();
        if (!codeformer_->Load(manifest_.codeformer, threads_, use_vulkan_, error))
            return false;
    }
    return true;
}

bool Pipeline::ensure_esrgan(std::string& error) {
    if (!esrgan_) {
        esrgan_ = std::make_unique<RealEsrgan>();
        if (!esrgan_->Load(manifest_.realesrgan, threads_, use_vulkan_, error))
            return false;
    }
    return true;
}

bool Pipeline::ensure_colorizer(std::string& error) {
    if (manifest_.colorize.param_path.empty() || manifest_.colorize.bin_path.empty()) {
        error = "colorization model is not available";
        return false;
    }
    if (!colorizer_) {
        colorizer_ = std::make_unique<Colorizer>();
        if (!colorizer_->Load(manifest_.colorize, threads_, error)) return false;
    }
    return true;
}

void Pipeline::paste_face(ImageRgb& canvas, int scale, const Face& face,
                          const ImageRgb& restored, const std::vector<float>& mask512) {
    const int s512 = manifest_.codeformer.size;
    float tpl[10];
    arcface_template_5pt(tpl, s512);

    Affine2 orig_to_512;
    if (!similarity_5pt(face.kps, tpl, orig_to_512)) return;

    // Canvas coordinates are original coordinates multiplied by scale:
    //   p512 = A * p_orig + t  and  p_canvas = scale * p_orig
    //   => p512 = (A / scale) * p_canvas + t
    // Only the linear part is divided by scale; the translation is unchanged.
    // Dividing it as well shifts every pasted face towards the origin.
    Affine2 canvas_to_512;
    double inv_s = 1.0 / scale;
    canvas_to_512.m[0] = orig_to_512.m[0] * inv_s;
    canvas_to_512.m[1] = orig_to_512.m[1] * inv_s;
    canvas_to_512.m[2] = orig_to_512.m[2];
    canvas_to_512.m[3] = orig_to_512.m[3] * inv_s;
    canvas_to_512.m[4] = orig_to_512.m[4] * inv_s;
    canvas_to_512.m[5] = orig_to_512.m[5];

    Affine2 back;
    if (!affine_invert(canvas_to_512, back)) return;

    // Destination ROI: where the four 512-square corners land in the canvas.
    double min_x = canvas.w, min_y = canvas.h, max_x = 0, max_y = 0;
    const double S = s512;
    const double corners[4][2] = {{0, 0}, {S, 0}, {S, S}, {0, S}};
    for (const auto& q : corners) {
        double cx = back.m[0] * q[0] + back.m[1] * q[1] + back.m[2];
        double cy = back.m[3] * q[0] + back.m[4] * q[1] + back.m[5];
        min_x = std::min(min_x, cx);
        min_y = std::min(min_y, cy);
        max_x = std::max(max_x, cx);
        max_y = std::max(max_y, cy);
    }
    int x0 = std::max(0, static_cast<int>(std::floor(min_x)));
    int y0 = std::max(0, static_cast<int>(std::floor(min_y)));
    int x1 = std::min(canvas.w, static_cast<int>(std::ceil(max_x)));
    int y1 = std::min(canvas.h, static_cast<int>(std::ceil(max_y)));

    const double* m = canvas_to_512.m;
    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            double qx = m[0] * x + m[1] * y + m[2];
            double qy = m[3] * x + m[4] * y + m[5];
            float alpha = static_cast<float>(sample_mask(mask512, s512, s512, qx, qy));
            if (alpha <= 0.f) continue;
            BilinearRgb sample = sample_rgb(restored, qx, qy);
            if (!sample.inside) continue;
            alpha = std::clamp(alpha, 0.f, 1.f);
            uint8_t* d = canvas.at(x, y);
            for (int c = 0; c < 3; ++c) {
                float v = alpha * sample.v[c] + (1.f - alpha) * d[c];
                d[c] = static_cast<uint8_t>(std::lround(v));
            }
        }
    }
}

bool Pipeline::Enhance(const uint8_t* src_rgba, int src_w, int src_h,
                       int src_stride, uint8_t* dst_rgba, int dst_capacity,
                       int mode, float fidelity, int target_scale,
                       int& out_w, int& out_h, std::string& error,
                       const ProgressFn& progress) {
    if (!src_rgba || !dst_rgba || src_w <= 0 || src_h <= 0) {
        error = "invalid input buffer";
        return false;
    }

    ImageRgb src;
    rgba_to_rgb(src_rgba, src_w, src_h, src_stride, src);

    // Colorization is a standalone transform: no detection, no upscaling.
    if (mode == IC_MODE_COLORIZE) {
        const auto t = Clock::now();
        if (progress) progress({IC_STAGE_COLORIZE, 0.f, "照片上色"});
        if (!ensure_colorizer(error)) return false;
        ImageRgb colored = colorizer_->Colorize(src);
        PIXELUP_LOGI("colorize: %lldms", ms_since(t));
        if (progress) progress({IC_STAGE_COLORIZE, 1.f, "照片上色"});
        out_w = colored.w;
        out_h = colored.h;
        if (static_cast<int64_t>(dst_capacity) <
            static_cast<int64_t>(out_w) * out_h * 4) {
            error = "output buffer too small";
            return false;
        }
        rgb_to_rgba(colored, dst_rgba, out_w * 4);
        return true;
    }

    const bool want_faces = mode != IC_MODE_UPSCALE;
    int scale = mode == IC_MODE_FACES ? 1 : (target_scale <= 1 ? 1 : target_scale < 3 ? 2 : 4);

    const auto t_total = Clock::now();

    std::vector<Face> faces;
    if (want_faces) {
        if (progress) progress({IC_STAGE_DETECT_FACES, 0.f, "检测人脸"});
        const auto t = Clock::now();
        if (!ensure_scrfd(error)) return false;
        faces = scrfd_->Detect(src);
        PIXELUP_LOGI("detect: %lldms (%zu faces)", ms_since(t), faces.size());
        if (progress) progress({IC_STAGE_DETECT_FACES, 1.f, "检测人脸"});
    }

    ImageRgb canvas;
    if (scale >= 2) {
        if (progress) progress({IC_STAGE_UPSCALE, 0.f, "超分辨率"});
        const auto t = Clock::now();
        if (!ensure_esrgan(error)) return false;
        ImageRgb up = esrgan_->Upscale(src, scale, [&](float p) {
            if (progress) progress({IC_STAGE_UPSCALE, p, "超分辨率"});
        });
        PIXELUP_LOGI("upscale: %lldms", ms_since(t));
        if (up.px.empty()) {
            error = "Real-ESRGAN produced no output";
            return false;
        }
        canvas = std::move(up);
        if (progress) progress({IC_STAGE_UPSCALE, 1.f, "超分辨率"});
    } else {
        canvas = src;
    }

    if (want_faces && !faces.empty()) {
        if (!ensure_codeformer(error)) return false;
        const int s512 = manifest_.codeformer.size;
        // Full frontal-head ellipse. The ArcFace template covers only
        // eyes-nose-mouth and its centre is high in the frame; centring the
        // mask there exposed the forehead/top-of-head, where the network's
        // letterboxed outside-face area is garbage (black/coloured blocks).
        std::vector<float> mask =
            ellipse_feather_mask(s512, s512 * 0.50f, s512 * 0.42f,
                                 s512 * 0.42f, s512 * 0.46f, s512 * 0.04f);
        for (size_t i = 0; i < faces.size(); ++i) {
            if (progress) {
                float p = static_cast<float>(i) / faces.size();
                progress({IC_STAGE_ENHANCE_FACES, p,
                          "修复人脸 " + std::to_string(i + 1) + "/" +
                              std::to_string(faces.size())});
            }
            float tpl[10];
            arcface_template_5pt(tpl, s512);
            Affine2 orig_to_512;
            if (!similarity_5pt(faces[i].kps, tpl, orig_to_512)) continue;
            Affine2 dst_to_src;
            if (!affine_invert(orig_to_512, dst_to_src)) continue;

            const auto t_face = Clock::now();
            ImageRgb aligned(s512, s512);
            warp_rgb_clamped(src, aligned, dst_to_src);
            ImageRgb restored = codeformer_->Restore(aligned, fidelity);
            PIXELUP_LOGI("face %zu/%zu: %lldms", i + 1, faces.size(), ms_since(t_face));

            // Only blend pixels whose alignment actually came from inside the
            // photo: the edge-clamped remainder is where CodeFormer invents
            // noisy garbage.
            std::vector<float> valid =
                warp_validity(src.w, src.h, s512, s512, dst_to_src, s512 * 0.012f);
            std::vector<float> face_mask(mask.size());
            for (size_t k = 0; k < mask.size(); ++k) {
                face_mask[k] = mask[k] * valid[k];
            }
            paste_face(canvas, scale, faces[i], restored, face_mask);
        }
        if (progress) progress({IC_STAGE_ENHANCE_FACES, 1.f, "人脸修复完成"});
    }

    out_w = canvas.w;
    out_h = canvas.h;
    if (static_cast<int64_t>(dst_capacity) <
        static_cast<int64_t>(out_w) * out_h * 4) {
        error = "output buffer too small";
        return false;
    }
    if (progress) progress({IC_STAGE_FINALIZE, 0.f, "生成结果"});
    rgb_to_rgba(canvas, dst_rgba, out_w * 4);
    PIXELUP_LOGI("total: %lldms (scale=%d, %zu faces)", ms_since(t_total), scale,
                 faces.size());
    if (progress) progress({IC_STAGE_FINALIZE, 1.f, "完成"});
    return true;
}

}  // namespace ic
