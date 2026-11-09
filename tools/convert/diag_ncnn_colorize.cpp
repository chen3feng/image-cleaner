// Runs the converted DDColor model on macOS ncnn for offline quality checks.
// Input is a raw planar float32 [1,3,256,256] (neutral-grey RGB in 0..1);
// output is a raw planar float32 [1,2,256,256] (Lab ab).
//
// build (from tools/convert):
//   xcrun --sdk macosx clang++ -std=c++17 -O2 diag_ncnn_colorize.cpp \
//       -I .cache/macncnn/ncnn.framework/Versions/A/Headers/ncnn \
//       -F .cache/macncnn -framework ncnn -framework openmp \
//       -o build/diag_ncnn_colorize
#include <cstdio>
#include <vector>

#include "net.h"

int main(int argc, char** argv) {
    if (argc != 5 && argc != 6) {
        std::fprintf(stderr, "usage: %s param bin in.bin out.bin [size]\n", argv[0]);
        return 2;
    }
    const int size = argc == 6 ? std::atoi(argv[5]) : 256;
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

    ncnn::Mat input(size, size, 3);
    FILE* f = std::fopen(argv[3], "rb");
    if (!f) return 1;
    const size_t want = (size_t)size * size * 3u;
    const size_t got = std::fread(input.data, sizeof(float), want, f);
    std::fclose(f);
    if (got != want) {
        std::fprintf(stderr, "short input %zu/%zu\n", got, want);
        return 1;
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.input("input", input);
    ncnn::Mat out;
    if (ex.extract("output", out) != 0) {
        std::fprintf(stderr, "extract failed\n");
        return 1;
    }
    std::printf("output %dx%dx%d\n", out.w, out.h, out.c);

    FILE* o = std::fopen(argv[4], "wb");
    if (!o) return 1;
    for (int c = 0; c < out.c; ++c) {
        for (int y = 0; y < out.h; ++y) {
            std::fwrite(out.channel(c).row(y), sizeof(float), out.w, o);
        }
    }
    std::fclose(o);
    return 0;
}
