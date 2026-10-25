#include "enhancer_c_api.h"

#include <memory>
#include <string>

#include "pipeline.h"

namespace {

thread_local std::string g_last_error;

struct Context {
    ic::Pipeline pipeline;
};

}  // namespace

extern "C" IC_Handle ic_create(const char* model_dir, const char* manifest_json,
                               int threads, int use_vulkan) {
    auto* ctx = new Context();
    std::string error;
    if (!ctx->pipeline.Init(model_dir ? model_dir : "",
                            manifest_json ? manifest_json : "",
                            threads, use_vulkan != 0, error)) {
        g_last_error = error;
        delete ctx;
        return nullptr;
    }
    return ctx;
}

extern "C" void ic_destroy(IC_Handle handle) {
    delete static_cast<Context*>(handle);
}

extern "C" int ic_enhance(IC_Handle handle,
                          const uint8_t* src_rgba, int src_w, int src_h,
                          int src_stride, uint8_t* dst_rgba, int dst_capacity,
                          int mode, float fidelity, int target_scale,
                          int* out_w, int* out_h,
                          IC_ProgressFn progress, void* user_data) {
    auto* ctx = static_cast<Context*>(handle);
    if (!ctx) {
        g_last_error = "null pipeline handle";
        return -1;
    }
    ic::ProgressFn cb;
    if (progress) {
        cb = [progress, user_data](const ic::Progress& p) {
            progress(p.stage, p.percent, p.message.c_str(), user_data);
        };
    }

    int w = 0, h = 0;
    std::string error;
    if (!ctx->pipeline.Enhance(src_rgba, src_w, src_h, src_stride, dst_rgba,
                               dst_capacity, mode, fidelity, target_scale,
                               w, h, error, cb)) {
        g_last_error = error;
        return -1;
    }
    if (out_w) *out_w = w;
    if (out_h) *out_h = h;
    return 0;
}

extern "C" int ic_inpaint(IC_Handle handle,
                          const uint8_t* src_rgba, int src_w, int src_h,
                          int src_stride, const uint8_t* mask, int mask_stride,
                          uint8_t* dst_rgba, int dst_capacity,
                          IC_ProgressFn progress, void* user_data) {
    auto* ctx = static_cast<Context*>(handle);
    if (!ctx) {
        g_last_error = "null pipeline handle";
        return -1;
    }
    ic::ProgressFn cb;
    if (progress) {
        cb = [progress, user_data](const ic::Progress& p) {
            progress(p.stage, p.percent, p.message.c_str(), user_data);
        };
    }
    std::string error;
    if (!ctx->pipeline.Inpaint(src_rgba, src_w, src_h, src_stride, mask,
                               mask_stride, dst_rgba, dst_capacity, error, cb)) {
        g_last_error = error;
        return -1;
    }
    return 0;
}

extern "C" const char* ic_last_error(void) {
    return g_last_error.c_str();
}
