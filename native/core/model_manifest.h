#pragma once

#include <string>

namespace ic {

struct ScrfdConfig {
    std::string param_path;
    std::string bin_path;
    std::string input_blob = "input";
    int input_size = 640;
    int strides[3] = {8, 16, 32};
    int num_anchors = 2;
    std::string score_blobs[3] = {"score_8", "score_16", "score_32"};
    std::string bbox_blobs[3] = {"bbox_8", "bbox_16", "bbox_32"};
    std::string kps_blobs[3] = {"kps_8", "kps_16", "kps_32"};
    float score_threshold = 0.5f;
    float nms_threshold = 0.4f;
    int max_faces = 10;
};

// CodeFormer ships as two graphs around the codebook lookup:
//   encoder: image (1,3,S,S) -> logits (1,256,1024), lq (1,256,16,16)
//   generator: quant (1,256,16,16) -> image (1,3,S,S)
// The hard argmax over logits + embedding gather + AdaIN run in C++, because
// ncnn's Vulkan 2D Reduction kernel corrupts that 256x1024 reduce on Adreno.
struct CodeFormerConfig {
    std::string enc_param_path;
    std::string enc_bin_path;
    std::string gen_param_path;
    std::string gen_bin_path;
    std::string codebook_path;      // raw float32 [codebook_size][emb_dim]
    std::string input_blob = "input";
    std::string logits_blob = "logits";
    std::string lq_blob = "lq";
    std::string gen_input_blob = "input";
    std::string output_blob = "output";
    std::string gen_output_blob = "output";
    int size = 512;
    int codebook_size = 1024;
    int emb_dim = 256;
    int latent_size = 16;           // 16x16 tokens
};

struct RealEsrganConfig {
    std::string param_path;
    std::string bin_path;
    std::string input_blob = "input";
    std::string output_blob = "output";
    int scale = 4;
    int tile = 256;
    int tile_overlap = 32;
};

struct ModelManifest {
    ScrfdConfig scrfd;
    CodeFormerConfig codeformer;
    RealEsrganConfig realesrgan;

    // Resolves every relative model path against model_dir and validates the
    // files exist. Returns false (with error set) otherwise.
    static bool Load(const std::string& model_dir, const std::string& json,
                     ModelManifest& out, std::string& error);
};

}  // namespace ic
