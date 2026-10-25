# PixelUp · 本地 AI 图像增强（image-cleaner）

Android 优先的手机端图像超分辨率 / 人脸修复 App。推理完全在本机运行（ncnn +
Vulkan），图片不上传服务器；推理核心是纯 C++/C ABI，未来可直接桥接到 iOS。

管线参考 [CodeFormer](https://github.com/sczhou/CodeFormer)，由三段组成：

```text
输入图片
   │
   ├── SCRFD 2.5g          人脸检测 + 5 点关键点（letterbox 640）
   │
   ├── Real-ESRGAN x4v3    全图超分（256 分块 + 羽化融合，可选 2x/4x）
   │
   └── CodeFormer          人脸对齐到 512² → transformer 修复 → 按关键点
                           逆变换贴回超分结果，椭圆羽化蒙版混合
   │
输出（可拖动分割线对比，保存到相册 / 分享）
```

## 特性

- **本地推理**：ncnn 1.0.20261020，Vulkan GPU 加速，FP16，无 GPU 时自动回退 CPU
- **高性能**：Direct ByteBuffer 零拷贝传图；模型按需懒加载；分块推理控制峰值内存；
  OpenMP 多线程
- **模型首启下载**：APK 保持小巧，SHA-256 校验，下载源走 GitHub Releases（可镜像）
- **内存预算**：4x 输入长边 ≤960、2x ≤1536，超大图自动等比缩小（见 ViewModel）
- **保真度滑块**：0 = 最强修复，1 = 保留原始纹理（客户端与 CodeFormer 官方一致的混合）

## 技术栈

| 层 | 选型 |
| --- | --- |
| UI | Kotlin + Jetpack Compose（Material 3） |
| 推理 | C++17 + [ncnn](https://github.com/Tencent/ncnn)（静态链接，Vulkan） |
| 桥接 | JNI（薄封装），对内暴露纯 C ABI `native/core/enhancer_c_api.h` |
| 模型 | SCRFD 2.5g / CodeFormer(fp16) / Real-ESRGAN general x4v3(fp16) |
| 下载 | OkHttp + 文件完整性校验 |

## 目录结构

```
app/                    Android 应用（Kotlin/Compose）
  src/main/assets/models/models.json   模型清单（下载项、哈希、ncnn blob 名）
native/
  CMakeLists.txt
  core/                 推理管线（与 Android 无关的纯 C++）
    enhancer_c_api.h    未来 iOS 也只依赖这个头
    pipeline.*          编排：检测 → 超分 → 人脸修复 → 贴回
    scrfd.* codeformer.* realesrgan.* image_utils.* model_manifest.*
    jni_bridge.cpp      Android 专用，仅此一个文件
  ncnn/                 预编译库（gitignore，setup_ncnn.sh 拉取）
tools/
  setup_ncnn.sh         拉取 ncnn Android Vulkan 预编译包
  convert/              PyTorch → ONNX/TorchScript → ncnn fp16 的全部脚本
  pack_models.sh        计算 size/SHA-256，更新清单，暂存 release 目录
  release_models.sh     gh CLI 发布模型 bundle（models-1 tag）
```

## 构建

需要：Android SDK Platform 36、NDK 29.0.14206865（ncnn 20261020 预编译包要求
r27+，r25 链接会报 `__libcpp_verbose_abort` 缺符号）、SDK CMake 3.10.2、JDK 17。

```bash
# 1. 拉取 ncnn 预编译库到 native/ncnn/（约 40 MB，不入 git）
tools/setup_ncnn.sh

# 2. 常规构建
./gradlew :app:assembleDebug
./gradlew :app:installDebug          # 需要 adb 连接设备
```

当前只编译 `arm64-v8a`（覆盖绝大多数现役设备）。要加 armeabi-v7a/x86_64：

```bash
ABIS="arm64-v8a armeabi-v7a x86_64" tools/setup_ncnn.sh
# 并把 ABI 加进 app/build.gradle.kts 的 ndk.abiFilters
```

## 模型准备（一次性，发布者操作）

模型权重**不随仓库分发**（体积 + 许可证原因）。流程：

```bash
cd tools/convert  # 以下命令都在这个目录下
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

mkdir weights
# 放入 codeformer.pth、realesr-general-x4v3.pth（链接见 convert_all.sh 注释）
# 另需本地有 CodeFormer / Real-ESRGAN 源码仓库（脚本默认路径可通过环境变量覆盖）

bash convert_all.sh                  # 产出 dist/ 下六个 .param/.bin
cd ../..
tools/pack_models.sh tools/convert/dist
tools/release_models.sh models-1     # gh release create
```

`pack_models.sh` 会把每个文件的大小和 SHA-256 写回
`app/src/main/assets/models/models.json`，并把 `base_url` 指向的 Release 产物
暂存到 `models-release/`。上架前把清单里的 `base_url` 改成自己的
`owner/repo`，或构建时覆盖：

```bash
./gradlew assembleRelease -PmodelsBaseUrl=https://example.com/models/
```

> 转换脚本里的 blob 重命名表按 insightface / pnnx 的常见输出命名；如果换了
> 转换工具版本，用 Netron 核对输出名，改 `rename_ncnn_blobs.py` 的 `--rename`
> 即可——native 侧所有 blob 名都来自 models.json，不需要重新编译 C++。

## 模型许可证（重要）

App 代码为 Apache-2.0，但模型权重各有许可，商用前务必确认：

- **CodeFormer**：NTU S-Lab License 1.0，**商业使用需授权**
- **Real-ESRGAN**：BSD 3-Clause
- **SCRFD（insightface buffalo_sc）**：MIT（insightface 模型库）

## iOS 支持：需要哪些牺牲 / 成本？

核心管线（`native/core/`，约 90% 的复杂代码）和 ncnn 本身在 iOS 上**原样复用**。
需要付出的代价集中在工程外壳：

1. **UI 全部重写**：Compose → SwiftUI，约是 App 层再写一遍（选择原生 Android
   先行时就接受的成本；若当初选 Flutter 可省掉这一步，但 FFI 插件构建更折腾）。
   需要一个 ~200 行的 ObjC++ wrapper 替换 `jni_bridge.cpp`（调用同一个
   `enhancer_c_api.h`）。
2. **GPU 后端不同**：Android 走 Vulkan，iOS 上 ncnn 自动改用 **Metal**
   （`opt.use_metal_compute`），代码一致但性能特征不同，要分别做 profiling
   和 tile/线程调参。
3. **平台 API 替换**：MediaStore → PhotoKit、FileProvider →
   UIActivityViewController、模型下载 OkHttp → URLSession（这层 Kotlin 不能复用）。
4. **分发成本**：Apple Developer 账号 \$99/年；审核周期；TestFlight；模型首启
   下载符合 Apple 政策（模型不是可执行代码），但要在审核说明里讲清楚。
5. **内存策略更保守**：iOS 的 jetsam 对单进程内存很严格，老设备建议沿用本项目的
   输入边长上限（4x 960 / 2x 1536）甚至再降一档。
6. **CI**：双端构建本来就必须在 macOS 上（当前环境已具备 Xcode）。

不存在需要放弃的功能或性能；主要“牺牲”就是 **App 外壳代码写两遍 + 双端分别调优
GPU 行为 + 苹果生态的分发门槛**。如果后续验证 iOS 是硬需求，建议尽早把
相册/分享/下载等平台操作收敛到一个 Kotlin/Swift 各自实现的薄接口后面。

## 路线图

- [ ] 真机验证模型转换产物（当前管线按 ncnn 标准 blob 约定实现，待权重转换后联调）
- [ ] armeabi-v7a / x86_64 ABI
- [ ] 端上 benchmark 页面（各阶段耗时、GPU/CPU 开关）
- [ ] 动漫模型（Real-ESRGAN animevideov3）、老照片去噪/上色
- [ ] iOS SwiftUI 壳工程
