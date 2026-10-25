#include "codeformer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <android/log.h>
#include <chrono>

namespace ic {
namespace {

void configure_net(ncnn::Net& net, int threads, bool use_vulkan) {
    net.opt.use_vulkan_compute = use_vulkan;
    net.opt.num_threads = threads;
    // The codebook selection now happens in C++, so the graphs no longer carry
    // the exp((p-max)*1e9) hardening that fp16 arithmetic used to overflow.
    // fp16 shaders are ~2.6x faster here with no visible accuracy loss
    // (logits within 0.06 of the fp32 run).
    net.opt.use_fp16_packed = use_vulkan;
    net.opt.use_fp16_storage = use_vulkan;
    net.opt.use_fp16_arithmetic = use_vulkan;
}

bool load_net(ncnn::Net& net, const std::string& param, const std::string& bin,
              const char* name, std::string& error) {
    if (net.load_param(param.c_str()) != 0) {
        error = std::string(name) + " load_param failed: " + param;
        return false;
    }
    if (net.load_model(bin.c_str()) != 0) {
        error = std::string(name) + " load_model failed: " + bin;
        return false;
    }
    return true;
}

// Per-channel mean / biased-or-unbiased std over a [c][h][w] float grid.
void channel_stats(const ncnn::Mat& m, std::vector<float>& mean,
                   std::vector<float>& stddev) {
    const int n = m.w * m.h;
    mean.resize(m.c);
    stddev.resize(m.c);
    for (int c = 0; c < m.c; ++c) {
        const float* p = m.channel(c);
        double sum = 0.0;
        for (int i = 0; i < n; ++i) sum += p[i];
        double mu = sum / n;
        double var = 0.0;
        for (int i = 0; i < n; ++i) {
            double d = p[i] - mu;
            var += d * d;
        }
        // Matches torch.var(dim=...) default (unbiased, divide by n-1).
        var = n > 1 ? var / (n - 1) : 0.0;
        mean[c] = static_cast<float>(mu);
        stddev[c] = static_cast<float>(std::sqrt(var + 1e-5));
    }
}

}  // namespace

bool CodeFormer::Load(const CodeFormerConfig& cfg, int threads, bool use_vulkan,
                      std::string& error) {
    cfg_ = cfg;
    configure_net(enc_, threads, use_vulkan);
    configure_net(gen_, threads, use_vulkan);

    if (!load_net(enc_, cfg.enc_param_path, cfg.enc_bin_path, "CodeFormer encoder", error))
        return false;
    if (!load_net(gen_, cfg.gen_param_path, cfg.gen_bin_path, "CodeFormer generator", error))
        return false;
    __android_log_print(ANDROID_LOG_INFO, "PixelUpNative",
                        "codeformer: vulkan=%d gpu_count=%d", use_vulkan ? 1 : 0,
                        ncnn::get_gpu_count());

    const size_t count = static_cast<size_t>(cfg.codebook_size) * cfg.emb_dim;
    codebook_.resize(count);
    FILE* f = std::fopen(cfg.codebook_path.c_str(), "rb");
    if (!f) {
        error = "codebook open failed: " + cfg.codebook_path;
        return false;
    }
    const size_t read = std::fread(codebook_.data(), sizeof(float), count, f);
    std::fclose(f);
    if (read != count) {
        error = "codebook read failed: " + cfg.codebook_path;
        return false;
    }
    loaded_ = true;
    return true;
}

ImageRgb CodeFormer::Restore(const ImageRgb& aligned, float fidelity) const {
    const int s = cfg_.size;
    const int lat = cfg_.latent_size;
    const int dim = cfg_.emb_dim;
    ImageRgb result(s, s);
    if (!loaded_ || aligned.w != s || aligned.h != s) return result;

    // ---------------- encoder: image -> logits, lq ----------------
    ncnn::Mat input = ncnn::Mat::from_pixels(
        aligned.px.data(), ncnn::Mat::PIXEL_RGB, s, s, s * 3);
    const float mean[3] = {127.5f, 127.5f, 127.5f};
    const float norm[3] = {1.0f / 127.5f, 1.0f / 127.5f, 1.0f / 127.5f};
    input.substract_mean_normalize(mean, norm);

    ncnn::Mat logits, lq;
    {
        const auto t = std::chrono::steady_clock::now();
        ncnn::Extractor ex = enc_.create_extractor();
        ex.set_light_mode(true);
        ex.input(cfg_.input_blob.c_str(), input);
        const int rc1 = ex.extract(cfg_.logits_blob.c_str(), logits);
        const int rc2 = ex.extract(cfg_.lq_blob.c_str(), lq);
        long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - t).count();
        __android_log_print(ANDROID_LOG_INFO, "PixelUpNative",
                            "  encoder: %lldms (rc=%d,%d)", ms, rc1, rc2);
        if (rc1 != 0 || rc2 != 0) {
            return aligned;
        }
    }
    const int tokens = lat * lat;
    if (logits.w != cfg_.codebook_size || logits.h != tokens ||
        lq.w != lat || lq.h != lat || lq.c != dim) {
        return aligned;  // model/blob mismatch; fail safe to the input face
    }

    // ---------------- hard codebook assignment + AdaIN ----------------
    // quant[c][y][x] = codebook[argmax_t logits[t][c]][c], t = y*lat + x
    ncnn::Mat quant(lat, lat, dim);
    std::vector<float> content_mean, content_std;
    for (int y = 0; y < lat; ++y) {
        for (int x = 0; x < lat; ++x) {
            const int t = y * lat + x;
            const float* row = logits.row(t);  // codebook_size scores
            int best = 0;
            float best_v = row[0];
            for (int k = 1; k < cfg_.codebook_size; ++k) {
                if (row[k] > best_v) {
                    best_v = row[k];
                    best = k;
                }
            }
            const float* emb = codebook_.data() + static_cast<size_t>(best) * dim;
            for (int c = 0; c < dim; ++c) {
                quant.channel(c).row(y)[x] = emb[c];
            }
        }
    }

    std::vector<float> style_mean, style_std;
    channel_stats(quant, content_mean, content_std);
    channel_stats(lq, style_mean, style_std);

    ncnn::Mat normalized(lat, lat, dim);
    for (int c = 0; c < dim; ++c) {
        const float* src = quant.channel(c);
        float* dst = normalized.channel(c);
        const float cm = content_mean[c], cs = content_std[c];
        const float sm = style_mean[c], ss = style_std[c];
        for (int i = 0; i < lat * lat; ++i) {
            dst[i] = (src[i] - cm) / cs * ss + sm;
        }
    }

    // ---------------- generator: quant -> image ----------------
    ncnn::Mat out;
    {
        const auto t = std::chrono::steady_clock::now();
        ncnn::Extractor ex = gen_.create_extractor();
        ex.set_light_mode(true);
        ex.input(cfg_.gen_input_blob.c_str(), normalized);
        const int rc = ex.extract(cfg_.gen_output_blob.c_str(), out);
        long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - t).count();
        __android_log_print(ANDROID_LOG_INFO, "PixelUpNative", "  generator: %lldms (rc=%d)",
                            ms, rc);
        if (rc != 0 || out.w != s || out.h != s || out.c < 3) {
            return aligned;
        }
    }

    const float wgt = std::min(1.f, std::max(0.f, fidelity));
    for (int c = 0; c < 3; ++c) {
        const float* p = out.channel(c);
        for (int y = 0; y < s; ++y) {
            for (int x = 0; x < s; ++x) {
                const float restored_raw = p[y * s + x] * 127.5f + 127.5f;
                const float restored = std::min(255.f, std::max(0.f, restored_raw));
                // Fidelity blend (graph is the w=0/max-restoration branch):
                // out = w * restored + (1 - w) * aligned_input
                const float v = wgt * restored +
                                (1.0f - wgt) * static_cast<float>(aligned.at(x, y)[c]);
                result.at(x, y)[c] = static_cast<uint8_t>(std::lround(
                    std::min(255.f, std::max(0.f, v))));
            }
        }
    }
    return result;
}

}  // namespace ic
