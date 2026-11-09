// Runs the native Colorizer on macOS for offline verification of the tiling.
// build (from tools/convert):
//   xcrun --sdk macosx clang++ -std=c++17 -O2 diag_colorize_cpp.cpp \
//       ../../native/core/colorize.cpp ../../native/core/image_utils.cpp \
//       -I ../../native/core \
//       -I .cache/macncnn/ncnn.framework/Versions/A/Headers/ncnn \
//       -F .cache/macncnn -framework ncnn -framework openmp \
//       -o build/diag_colorize_cpp
#include <cstdio>
#include <cstdlib>
#include <string>

#include "colorize.h"

int main(int argc, char** argv) {
    if (argc != 7 && argc != 8) {
        std::fprintf(stderr,
                     "usage: %s param bin in.raw w h out.raw [strength]\n", argv[0]);
        return 2;
    }
    ic::ColorizeConfig cfg;
    cfg.param_path = argv[1];
    cfg.bin_path = argv[2];
    ic::Colorizer colorizer;
    std::string error;
    if (!colorizer.Load(cfg, 4, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }

    const int w = std::atoi(argv[4]);
    const int h = std::atoi(argv[5]);
    ic::ImageRgb img(w, h);
    FILE* f = std::fopen(argv[3], "rb");
    if (!f || std::fread(img.px.data(), 1, img.px.size(), f) != img.px.size()) {
        std::fprintf(stderr, "bad input\n");
        return 1;
    }
    std::fclose(f);

    const float strength = argc == 8 ? std::atof(argv[7]) : 1.f;
    auto out = colorizer.Colorize(img, strength, [](float p) {
        std::fprintf(stderr, "progress %.2f\n", p);
    });
    FILE* g = std::fopen(argv[6], "wb");
    if (!g) return 1;
    std::fwrite(out.px.data(), 1, out.px.size(), g);
    std::fclose(g);
    return 0;
}
