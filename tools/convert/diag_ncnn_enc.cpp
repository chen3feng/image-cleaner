// Runs the converted CodeFormer encoder on macOS ncnn and dumps logits/lq so
// they can be compared with the PyTorch reference (see diag_encoder.py).
//
// build (from tools/convert):
//   c++ -std=c++17 -O2 diag_ncnn_enc.cpp \
//       -I .cache/macncnn/ncnn.framework/Versions/A/Headers \
//       -F .cache/macncnn -framework ncnn -framework openmp \
//       -o build/diag_ncnn_enc
//
// run:
//   build/diag_ncnn_enc enc.param enc.bin input.bin logits_ncnn.bin lq_ncnn.bin
#include <cstdio>
#include <cstring>
#include <vector>

#include "net.h"

static bool dump_mat(const char* path, const ncnn::Mat& m) {
    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    std::vector<float> row(m.w);
    for (int c = 0; c < m.c; ++c) {
        for (int y = 0; y < m.h; ++y) {
            const float* p = m.channel(c).row(y);
            std::fwrite(p, sizeof(float), m.w, f);
        }
    }
    std::fclose(f);
    return true;
}

int main(int argc, char** argv) {
    if (argc != 6 && argc != 9) {
        std::fprintf(stderr,
                     "usage: %s param bin input.bin logits.bin lq.bin "
                     "[in_blob logits_blob lq_blob]\n", argv[0]);
        return 2;
    }
    const char* in_blob = argc == 9 ? argv[6] : "input";
    const char* logits_blob = argc == 9 ? argv[7] : "logits";
    const char* lq_blob = argc == 9 ? argv[8] : "lq";
    ncnn::Net net;
    net.opt.num_threads = 4;
    net.opt.use_vulkan_compute = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    if (net.load_param(argv[1]) != 0 || net.load_model(argv[2]) != 0) {
        std::fprintf(stderr, "failed to load model\n");
        return 1;
    }

    ncnn::Mat input(512, 512, 3);
    FILE* f = std::fopen(argv[3], "rb");
    if (!f) {
        std::fprintf(stderr, "cannot open input\n");
        return 1;
    }
    const size_t want = 512u * 512u * 3u;
    const size_t got = std::fread(input.data, sizeof(float), want, f);
    std::fclose(f);
    if (got != want) {
        std::fprintf(stderr, "short input: %zu/%zu\n", got, want);
        return 1;
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.input(in_blob, input);
    ncnn::Mat logits, lq;
    int rc1 = ex.extract(logits_blob, logits);
    int rc2 = ex.extract(lq_blob, lq);
    std::printf("extract rc=%d,%d logits=%dx%dx%d lq=%dx%dx%d\n", rc1, rc2,
                logits.w, logits.h, logits.c, lq.w, lq.h, lq.c);
    if (rc1 != 0 || rc2 != 0) return 1;

    dump_mat(argv[4], logits);
    dump_mat(argv[5], lq);
    return 0;
}
