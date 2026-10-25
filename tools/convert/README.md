# 模型转换

C++ 管线不写死任何 blob 名，全部来自 `app/src/main/assets/models/models.json`。
转换的核心工作就是把转换器产出的 IO 名改成清单里的约定名。

## 约定

| 模型 | 输入 | 输出 |
|---|---|---|
| SCRFD 2.5g | `input` | `score_8/16/32`、`bbox_8/16/32`、`kps_8/16/32`（9 个） |
| CodeFormer | `input`（1×3×512×512，[-1,1]） | `output`（1×3×512×512，[-1,1]） |
| Real-ESRGAN x4v3 | `input`（[0,1] RGB，任意 256 tile） | `output`（4x） |

## 步骤

```bash
convert_all.sh
# 1) 下载 ncnn 工具（onnx2ncnn / ncnnoptimize）与 pnnx
# 2) SCRFD: buffalo_sc.zip → det_2.5g.onnx → onnx2ncnn → 改名 → fp16
# 3) CodeFormer: codeformer.pth → TorchScript → pnnx → 改名 → fp16
# 4) Real-ESRGAN: .pth → ONNX → onnx2ncnn → 改名 → fp16
```

## 转换后必须核对

1. `ncnnoptimize` 无 unsupported op 警告；如有，优先换 pnnx 路线。
2. 用 Netron 打开每个 `.param`：
   - 第一层 `Input` 名与上表一致；
   - SCRFD 九个输出名与 stride 对得上（8/16/32，每 stride 两组 anchor）；
   - CodeFormer 输入固定 512、无动态维。
3. SCRFD 输入尺寸：det_2.5g 通常是 640；若 Netron 显示 320，修改 manifest
   的 `input_size`（native 不需要重编译，但要重新发版 APK）。
4. fp16 后 CodeFormer 人脸不应有明显色块；若有问题，对该模型单独跑
   `ncnnoptimize` flag=0（fp32）再观察对比。
