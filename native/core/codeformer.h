#pragma once

#include <string>
#include <vector>

#include "image_utils.h"
#include "model_manifest.h"
#include "ncnn/net.h"

namespace ic {

// CodeFormer w=0 / adain=True restoration, split around the codebook lookup:
//   encoder (ncnn, GPU) -> logits + lq
//   C++: hard argmax -> codebook gather -> AdaIN
//   generator (ncnn, GPU) -> restored face
// The split exists because ncnn's Vulkan 2D Reduction on the 256x1024 logits
// matrix returns garbage on Adreno; doing the selection in C++ keeps every
// remaining operator on a well-tested kernel path.
class CodeFormer {
public:
    bool Load(const CodeFormerConfig& cfg, int threads, bool use_vulkan,
              std::string& error);

    // aligned: size x size RGB face produced with the ArcFace template.
    // fidelity in [0,1]; 0 = maximum restoration, 1 = keep original texture.
    ImageRgb Restore(const ImageRgb& aligned, float fidelity) const;

private:
    ncnn::Net enc_;
    ncnn::Net gen_;
    CodeFormerConfig cfg_;
    std::vector<float> codebook_;  // [codebook_size][emb_dim]
    bool loaded_ = false;
};

}  // namespace ic
