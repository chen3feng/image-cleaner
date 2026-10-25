#include "model_manifest.h"

#include <sys/stat.h>

#include "json_min.h"

namespace ic {
namespace {

std::string join_path(const std::string& dir, const std::string& name) {
    if (dir.empty() || name.empty()) return name;
    if (dir.back() == '/') return dir + name;
    return dir + "/" + name;
}

bool file_exists(const std::string& path) {
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

std::string array_string_at(const JsonValue* arr, int index,
                            const std::string& fallback) {
    if (!arr || arr->type != JsonValue::Array || index < 0 ||
        static_cast<size_t>(index) >= arr->arr.size()) {
        return fallback;
    }
    const JsonValue& item = arr->arr[index];
    return item.type == JsonValue::String ? item.str : fallback;
}

void resolve_weights(const JsonValue& section, const std::string& model_dir,
                     std::string& param_path, std::string& bin_path) {
    param_path = join_path(model_dir, section.get_string("param"));
    bin_path = join_path(model_dir, section.get_string("bin"));
}

}  // namespace

bool ModelManifest::Load(const std::string& model_dir, const std::string& json,
                         ModelManifest& out, std::string& error) {
    JsonValue root;
    if (!json_parse(json, root, error)) return false;

    const JsonValue* s = root.find("scrfd");
    const JsonValue* c = root.find("codeformer");
    const JsonValue* r = root.find("realesrgan");
    if (!s || !c || !r) {
        error = "manifest is missing one of: scrfd/codeformer/realesrgan";
        return false;
    }

    // SCRFD
    {
        const JsonValue& v = *s;
        resolve_weights(v, model_dir, out.scrfd.param_path, out.scrfd.bin_path);
        out.scrfd.input_blob = v.get_string("input_blob", out.scrfd.input_blob);
        out.scrfd.input_size = v.get_int("input_size", out.scrfd.input_size);
        out.scrfd.num_anchors = v.get_int("num_anchors", out.scrfd.num_anchors);
        out.scrfd.score_threshold =
            static_cast<float>(v.get_number("score_threshold", out.scrfd.score_threshold));
        out.scrfd.nms_threshold =
            static_cast<float>(v.get_number("nms_threshold", out.scrfd.nms_threshold));
        out.scrfd.max_faces = v.get_int("max_faces", out.scrfd.max_faces);

        if (const JsonValue* strides = v.find("strides")) {
            for (int i = 0; i < 3; ++i) {
                if (i < static_cast<int>(strides->arr.size()) &&
                    strides->arr[i].type == JsonValue::Number) {
                    out.scrfd.strides[i] =
                        static_cast<int>(strides->arr[i].number);
                }
            }
        }
        for (int i = 0; i < 3; ++i) {
            out.scrfd.score_blobs[i] =
                array_string_at(v.find("score_blobs"), i, out.scrfd.score_blobs[i]);
            out.scrfd.bbox_blobs[i] =
                array_string_at(v.find("bbox_blobs"), i, out.scrfd.bbox_blobs[i]);
            out.scrfd.kps_blobs[i] =
                array_string_at(v.find("kps_blobs"), i, out.scrfd.kps_blobs[i]);
        }
    }

    // CodeFormer (split encoder / generator + external codebook)
    {
        const JsonValue& v = *c;
        resolve_weights(v, model_dir, out.codeformer.enc_param_path,
                        out.codeformer.enc_bin_path);
        const JsonValue* gen = v.find("generator");
        if (!gen) {
            error = "manifest: codeformer.generator missing";
            return false;
        }
        resolve_weights(*gen, model_dir, out.codeformer.gen_param_path,
                        out.codeformer.gen_bin_path);
        out.codeformer.codebook_path =
            join_path(model_dir, v.get_string("codebook"));
        out.codeformer.input_blob = v.get_string("input_blob", out.codeformer.input_blob);
        out.codeformer.logits_blob = v.get_string("logits_blob", out.codeformer.logits_blob);
        out.codeformer.lq_blob = v.get_string("lq_blob", out.codeformer.lq_blob);
        out.codeformer.gen_input_blob =
            gen->get_string("input_blob", out.codeformer.gen_input_blob);
        out.codeformer.gen_output_blob =
            gen->get_string("output_blob", out.codeformer.gen_output_blob);
        out.codeformer.size = v.get_int("size", out.codeformer.size);
        out.codeformer.codebook_size = v.get_int("codebook_size", out.codeformer.codebook_size);
        out.codeformer.emb_dim = v.get_int("emb_dim", out.codeformer.emb_dim);
        out.codeformer.latent_size = v.get_int("latent_size", out.codeformer.latent_size);
    }

    // Real-ESRGAN
    {
        const JsonValue& v = *r;
        resolve_weights(v, model_dir, out.realesrgan.param_path, out.realesrgan.bin_path);
        out.realesrgan.input_blob = v.get_string("input_blob", out.realesrgan.input_blob);
        out.realesrgan.output_blob = v.get_string("output_blob", out.realesrgan.output_blob);
        out.realesrgan.scale = v.get_int("scale", out.realesrgan.scale);
        out.realesrgan.tile = v.get_int("tile", out.realesrgan.tile);
        out.realesrgan.tile_overlap = v.get_int("tile_overlap", out.realesrgan.tile_overlap);
    }

    const std::string* paths[] = {
        &out.scrfd.param_path, &out.scrfd.bin_path,
        &out.codeformer.enc_param_path, &out.codeformer.enc_bin_path,
        &out.codeformer.gen_param_path, &out.codeformer.gen_bin_path,
        &out.codeformer.codebook_path,
        &out.realesrgan.param_path, &out.realesrgan.bin_path};
    for (const std::string* p : paths) {
        if (!file_exists(*p)) {
            error = "model file missing: " + *p;
            return false;
        }
    }

    struct stat st {};
    if (::stat(out.codeformer.codebook_path.c_str(), &st) != 0 ||
        st.st_size != static_cast<off_t>(out.codeformer.codebook_size) *
                          out.codeformer.emb_dim * 4) {
        error = "codebook size mismatch (expected " +
                std::to_string(out.codeformer.codebook_size * out.codeformer.emb_dim * 4) +
                " bytes): " + out.codeformer.codebook_path;
        return false;
    }
    return true;
}

}  // namespace ic
