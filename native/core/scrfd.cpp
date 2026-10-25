#include "scrfd.h"

#include <algorithm>
#include <cmath>

#include <android/log.h>

namespace ic {
namespace {

// NOTE: the insightface det_10g export already ends its three score branches
// with Sigmoid nodes (verified in the ONNX graph), so the blobs are
// probabilities. Applying sigmoid again mapped background anchors (raw 0) to
// 0.5, which passed the 0.5 threshold: ~8.8k candidates instead of a few dozen
// and an O(n^2) NMS that dominated the runtime.

float iou(const Face& a, const Face& b) {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);
    float inter_w = std::max(0.f, x2 - x1);
    float inter_h = std::max(0.f, y2 - y1);
    float inter = inter_w * inter_h;
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (area_a + area_b - inter + 1e-6f);
}

}  // namespace

bool Scrfd::Load(const ScrfdConfig& cfg, int threads, bool use_vulkan,
                 std::string& error) {
    cfg_ = cfg;
    net_.opt.use_vulkan_compute = use_vulkan;
    net_.opt.num_threads = threads;
    net_.opt.use_fp16_packed = true;
    net_.opt.use_fp16_storage = true;
    net_.opt.use_fp16_arithmetic = use_vulkan;

    if (net_.load_param(cfg.param_path.c_str()) != 0) {
        error = "SCRFD load_param failed: " + cfg.param_path;
        return false;
    }
    if (net_.load_model(cfg.bin_path.c_str()) != 0) {
        error = "SCRFD load_model failed: " + cfg.bin_path;
        return false;
    }
    loaded_ = true;
    return true;
}

std::vector<Face> Scrfd::Detect(const ImageRgb& image) const {
    std::vector<Face> empty;
    if (!loaded_) return empty;

    const int input_size = cfg_.input_size;
    const float ratio = std::min(static_cast<float>(input_size) / image.w,
                                 static_cast<float>(input_size) / image.h);
    const int new_w = std::max(1, static_cast<int>(std::round(image.w * ratio)));
    const int new_h = std::max(1, static_cast<int>(std::round(image.h * ratio)));

    ncnn::Mat resized;
    {
        ncnn::Mat tmp = ncnn::Mat::from_pixels_resize(
            image.px.data(), ncnn::Mat::PIXEL_RGB, image.w, image.h,
            image.w * 3, new_w, new_h);
        // Bottom/right letterbox padding to input_size x input_size
        resized = ncnn::Mat(input_size, input_size, 3);
        resized.fill(0.f);
        for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < new_h; ++y) {
                const float* src_row = tmp.channel(c).row<const float>(y);
                float* dst_row = resized.channel(c).row<float>(y);
                std::copy(src_row, src_row + new_w, dst_row);
            }
        }
    }

    const float mean[3] = {127.5f, 127.5f, 127.5f};
    const float norm[3] = {1.0f / 128.0f, 1.0f / 128.0f, 1.0f / 128.0f};
    resized.substract_mean_normalize(mean, norm);

    ncnn::Extractor ex = net_.create_extractor();
    ex.set_light_mode(true);
    ex.input(cfg_.input_blob.c_str(), resized);

    struct Candidate {
        Face face;
    };
    std::vector<Face> candidates;

    for (int s = 0; s < 3; ++s) {
        ncnn::Mat scores, bboxes, kps;
        if (ex.extract(cfg_.score_blobs[s].c_str(), scores) != 0 ||
            ex.extract(cfg_.bbox_blobs[s].c_str(), bboxes) != 0 ||
            ex.extract(cfg_.kps_blobs[s].c_str(), kps) != 0) {
            return empty;  // blob names in the manifest do not match the model
        }

        const int stride = cfg_.strides[s];
        const int grid = input_size / stride;
        const int anchors = cfg_.num_anchors;
        const int count = grid * grid * anchors;

        const auto* score_ptr = static_cast<const float*>(scores.data);
        const auto* bbox_ptr = static_cast<const float*>(bboxes.data);
        const auto* kps_ptr = static_cast<const float*>(kps.data);
        if (scores.w * scores.h * scores.c < count ||
            bboxes.w * bboxes.h * bboxes.c < count * 4 ||
            kps.w * kps.h * kps.c < count * 10) {
            return empty;
        }

        for (int k = 0; k < count; ++k) {
            float score = score_ptr[k];
            if (score < cfg_.score_threshold) continue;

            const int anchor = k % anchors;
            const int pos = k / anchors;
            const int cx_grid = pos % grid;
            const int cy_grid = pos / grid;
            (void)anchor;

            const float px = (cx_grid + 0.5f) * stride;
            const float py = (cy_grid + 0.5f) * stride;

            Face f{};
            f.score = score;
            f.x1 = px - bbox_ptr[k * 4 + 0] * stride;
            f.y1 = py - bbox_ptr[k * 4 + 1] * stride;
            f.x2 = px + bbox_ptr[k * 4 + 2] * stride;
            f.y2 = py + bbox_ptr[k * 4 + 3] * stride;
            for (int p = 0; p < 5; ++p) {
                f.kps[p * 2 + 0] = px + kps_ptr[k * 10 + p * 2 + 0] * stride;
                f.kps[p * 2 + 1] = py + kps_ptr[k * 10 + p * 2 + 1] * stride;
            }

            // Drop boxes centred in the letterbox padding.
            float mx = (f.x1 + f.x2) * 0.5f;
            float my = (f.y1 + f.y2) * 0.5f;
            if (mx <= 0 || my <= 0 || mx >= new_w || my >= new_h) continue;

            f.x1 /= ratio;
            f.y1 /= ratio;
            f.x2 /= ratio;
            f.y2 /= ratio;
            for (int p = 0; p < 5; ++p) {
                f.kps[p * 2 + 0] /= ratio;
                f.kps[p * 2 + 1] /= ratio;
            }
            f.x1 = std::max(0.f, std::min(f.x1, static_cast<float>(image.w - 1)));
            f.y1 = std::max(0.f, std::min(f.y1, static_cast<float>(image.h - 1)));
            f.x2 = std::max(0.f, std::min(f.x2, static_cast<float>(image.w - 1)));
            f.y2 = std::max(0.f, std::min(f.y2, static_cast<float>(image.h - 1)));
            if (f.x2 - f.x1 < 8.f || f.y2 - f.y1 < 8.f) continue;
            candidates.push_back(f);
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Face& a, const Face& b) { return a.score > b.score; });

#ifndef NDEBUG
    __android_log_print(ANDROID_LOG_DEBUG, "PixelUpNative",
                        "scrfd: %zu candidates above %.2f", candidates.size(),
                        cfg_.score_threshold);
#endif

    // Greedy NMS, single class.
    std::vector<char> removed(candidates.size(), 0);
    std::vector<Face> result;
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (removed[i]) continue;
        result.push_back(candidates[i]);
        if (static_cast<int>(result.size()) >= cfg_.max_faces) break;
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (!removed[j] && iou(candidates[i], candidates[j]) > cfg_.nms_threshold)
                removed[j] = 1;
        }
    }
    return result;
}

}  // namespace ic
