// Pure C ABI for the enhancement pipeline.
// The JNI layer (Android) and an ObjC++ bridge (future iOS) both call this,
// so nothing in this header may depend on ncnn, the STL ABI, or JNI.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* IC_Handle;

enum IC_Mode {
    IC_MODE_AUTO = 0,      // face restoration + whole-image super-resolution
    IC_MODE_FACES = 1,     // CodeFormer on faces only, original resolution kept
    IC_MODE_UPSCALE = 2,   // Real-ESRGAN only, no face detection
    IC_MODE_COLORIZE = 3   // DDColor: colourise a black-and-white photo
};

enum IC_Stage {
    IC_STAGE_PREPARE = 0,
    IC_STAGE_DETECT_FACES = 1,
    IC_STAGE_UPSCALE = 2,
    IC_STAGE_ENHANCE_FACES = 3,
    IC_STAGE_FINALIZE = 4,
    IC_STAGE_COLORIZE = 5
};

typedef void (*IC_ProgressFn)(int stage, float percent, const char* message,
                              void* user_data);

// model_dir: directory containing the .param/.bin files listed in the manifest.
// manifest_json: contents of models.json (blob names and runtime parameters).
// Returns NULL on failure; call ic_last_error().
IC_Handle ic_create(const char* model_dir, const char* manifest_json,
                    int threads, int use_vulkan);

void ic_destroy(IC_Handle handle);

// Pixels are tightly packed RGBA (Android Bitmap ARGB_8888 byte order).
// dst is caller-allocated with capacity dst_capacity_bytes; the produced frame
// is *out_w * *out_h * 4, where dims = src * (mode == FACES ? 1 : target_scale).
// Returns 0 on success, negative on failure (see ic_last_error()).
int ic_enhance(IC_Handle handle,
               const uint8_t* src_rgba, int src_w, int src_h, int src_stride,
               uint8_t* dst_rgba, int dst_capacity_bytes,
               int mode, float fidelity, int target_scale,
               int* out_w, int* out_h,
               IC_ProgressFn progress, void* user_data);

// Thread-local, never NULL.
const char* ic_last_error(void);

#ifdef __cplusplus
}
#endif
