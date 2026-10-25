#include <jni.h>

#include <cstdint>
#include <cstring>

#include "enhancer_c_api.h"

namespace {

JavaVM* g_vm = nullptr;
jclass g_native_class = nullptr;       // global ref
// Resolved lazily: during JNI_OnLoad the Kotlin object is still running its
// <clinit> (that is what called loadLibrary), and ART rejects GetStaticMethodID
// with NoSuchMethodError at that point. By the first progress callback the
// class has finished initializing.
jmethodID g_dispatch_progress = nullptr;

struct ProgressContext {
    jlong callback_id;
};

void progress_trampoline(int stage, float percent, const char* message,
                         void* user_data) {
    if (!g_vm || !g_dispatch_progress || !user_data) return;
    auto* ctx = static_cast<ProgressContext*>(user_data);

    JNIEnv* env = nullptr;
    bool attached = false;
    if (g_vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
        if (g_vm->AttachCurrentThread(&env, nullptr) != JNI_OK) return;
        attached = true;
    }

    if (!g_dispatch_progress) {
        g_dispatch_progress = env->GetStaticMethodID(
            g_native_class, "dispatchProgress", "(JFILjava/lang/String;)V");
        if (!g_dispatch_progress) {
            env->ExceptionClear();
            if (attached) g_vm->DetachCurrentThread();
            return;
        }
    }
    jstring msg = env->NewStringUTF(message ? message : "");
    env->CallStaticVoidMethod(g_native_class, g_dispatch_progress,
                              ctx->callback_id, static_cast<jint>(stage),
                              static_cast<jfloat>(percent), msg);
    if (env->ExceptionCheck()) env->ExceptionClear();
    env->DeleteLocalRef(msg);
    if (attached) g_vm->DetachCurrentThread();
}

}  // namespace

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
    g_vm = vm;
    JNIEnv* env = nullptr;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK)
        return JNI_ERR;
    jclass local = env->FindClass("com/imagecleaner/app/core/NativeEnhancer");
    if (!local) return JNI_ERR;
    g_native_class = reinterpret_cast<jclass>(env->NewGlobalRef(local));
    if (!g_native_class) {
        env->ExceptionClear();
        return JNI_ERR;
    }
    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_imagecleaner_app_core_NativeEnhancer_nativeCreate(
    JNIEnv* env, jclass, jstring model_dir, jstring manifest_json,
    jint threads, jboolean use_vulkan) {
    const char* dir = model_dir ? env->GetStringUTFChars(model_dir, nullptr) : "";
    const char* json =
        manifest_json ? env->GetStringUTFChars(manifest_json, nullptr) : "";
    IC_Handle handle = ic_create(dir, json, threads, use_vulkan == JNI_TRUE);
    if (model_dir) env->ReleaseStringUTFChars(model_dir, dir);
    if (manifest_json) env->ReleaseStringUTFChars(manifest_json, json);
    return reinterpret_cast<jlong>(handle);
}

extern "C" JNIEXPORT void JNICALL
Java_com_imagecleaner_app_core_NativeEnhancer_nativeDestroy(
    JNIEnv*, jclass, jlong handle) {
    ic_destroy(reinterpret_cast<IC_Handle>(handle));
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_imagecleaner_app_core_NativeEnhancer_nativeLastError(JNIEnv* env, jclass) {
    return env->NewStringUTF(ic_last_error());
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_imagecleaner_app_core_NativeEnhancer_nativeEnhance(
    JNIEnv* env, jclass, jlong handle, jobject src, jint src_w, jint src_h,
    jobject dst, jint dst_capacity, jint mode, jfloat fidelity, jint target_scale,
    jlong callback_id) {
    auto* h = reinterpret_cast<IC_Handle>(handle);
    auto* src_ptr = static_cast<uint8_t*>(env->GetDirectBufferAddress(src));
    auto* dst_ptr = static_cast<uint8_t*>(env->GetDirectBufferAddress(dst));
    if (!h || !src_ptr || !dst_ptr) return nullptr;

    ProgressContext ctx{callback_id};
    int out_w = 0, out_h = 0;
    int rc = ic_enhance(h, src_ptr, src_w, src_h, src_w * 4, dst_ptr,
                        dst_capacity, mode, fidelity, target_scale,
                        &out_w, &out_h,
                        callback_id != 0 ? progress_trampoline : nullptr, &ctx);
    if (rc != 0) return nullptr;

    jintArray dims = env->NewIntArray(2);
    jint buf[2] = {out_w, out_h};
    env->SetIntArrayRegion(dims, 0, 2, buf);
    return dims;
}
