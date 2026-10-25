# 模型转换

C++ 管线不写死任何 blob 名，全部来自 `app/src/main/assets/models/models.json`。
转换的核心工作就是让转换器产出的 IO 名与清单里的约定名一致。

## 约定

| 模型 | 输入 | 输出 |
| --- | --- | --- |
| SCRFD 10g | `input`（1×3×640×640，letterbox） | `score_8/16/32`、`bbox_8/16/32`、`kps_8/16/32`（9 个） |
| CodeFormer 编码器 | `input`（1×3×512×512，[-1,1]） | `logits`（1×256×1024）、`lq`（1×256×16×16） |
| CodeFormer 生成器 | `input`（1×256×16×16，AdaIN 后的码本特征） | `output`（1×3×512×512，[-1,1]） |
| Real-ESRGAN x4v3 | `input`（[0,1] RGB，任意 256 tile） | `output`（4x） |
| DDColor 上色 | `input`（1×3×256×256，[0,1] 的中性灰 RGB） | `output`（1×2×256×256，Lab ab） |

另有 `codeformer-codebook.bin`：码本权重 1024×256 的裸 float32（1 MB），
由 `export_codeformer.py` 一并导出。

## 为什么 CodeFormer 要拆成两段

原始导出的单图在 ncnn 上**能加载但结果不对**：

1. `nn.MultiheadAttention` 被 pnnx 降级成 ncnn 的 `MultiHeadAttention` 层后
   输出布局错误（单层实测：输入 256×512 出来 512×512×256）。
2. 码本硬选择需要在 256×1024 的 logits 上做 argmax。任何用
   `Reduction(max)` 表达的写法在 **Vulkan** 上都会算出 NaN/垃圾
   （`exp((p-max)*1e9)`、迭代平方归一化都试过；CPU 正常、GPU 全错）。
3. 换成显式 matmul/softmax 重写 attention 虽然与 PyTorch 完全一致，但 pnnx
   会漏出 ncnn 不认识的不透明节点（`nn.Linear`/`nn.LayerNorm`），模型无法加载。

所以最终方案把**码本 argmax + 查表 + AdaIN 放到 C++**
（`native/core/codeformer.cpp`），图上只保留卷积/transformer/生成器，
全部跑在 Vulkan 上。这样既精确（与 PyTorch 差 <2e-4）又快（fp16 shader）。

## DDColor 上色的两个坑

1. **Einsum**：`bqc,bchw->bqhw`（色度查询与像素特征的逐点内积）被 pnnx 转成
   ncnn 的 Einsum 层后，参数编码有误，设备上直接段错误。因为 B 本身就是
   `(b, c, h*w)`，等价的 `Reshape + MatMul + Reshape` 更稳，且与 ONNX 结果
   逐位相同（`export_colorize.py` 里有自检）。
2. **精度**：fp16 storage 会让 ab 明显漂移（平均误差 2.2，fp32 为 0.13），
   所以 DDColor 固定走 CPU fp32 —— 256×256 只要 0.8 秒，比 Vulkan 路径还快。

另外它的输入是 **RGB [0,1]**（不是 [-1,1]），ab 直接就是 Lab 色度（不要再乘 110）。

## 步骤

```bash
# 需要已安装 torch/onnx/pnnx 依赖的 venv
PYTHON=../.venv/bin/python bash convert_all.sh
```

流程：`codeformer.pth` → TorchScript（编码器/生成器各一份）→ pnnx（fp16）→
改名；SCRFD 从 insightface buffalo_l 取 `det_10g.onnx`，onnxsim 固定为
1×3×640×640 后交给 pnnx；Real-ESRGAN 导出 TorchScript 后 pnnx。

## 转换后必须核对

1. **pnnx 与 ncnn 运行时版本必须同期**：转换用 `PNNX_VERSION`，运行时用
   `tools/setup_ncnn.sh` 的 `NCNN_VERSION`，默认都是 `20260526`。旧版运行时
   解析新 pnnx 产物会在 `load_param` 里空指针崩溃。
2. 用 Netron / grep 检查 `.param`：
   - 不应出现 `nn.*` / `torch.*` / `aten::*` 等不透明层类型
     （`grep -E '^(nn\.|torch\.|aten::)' *.param` 应为空）；
   - SCRFD 九个输出按 stride 8/16/32 分组，每 stride 一组 score/bbox/kps；
   - CodeFormer 输入固定 512，无动态维。
3. **SCRFD 的 score 输出已内置 Sigmoid**（insightface 导出如此）。C++ 侧
   不要再做一次 sigmoid，否则背景 anchor（raw=0）会变成 0.5 全部过阈值。
4. SCRFD 输入尺寸见 manifest 的 `input_size`；换成 320 输入的模型时改清单即可，
   不必重编 native。
5. 精度回归：`export_codeformer.py` 末尾会打印 split 与官方路径的误差，
   应在 1e-4 量级；若明显变大，说明 pnnx 降级出了问题。
