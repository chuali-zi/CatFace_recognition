# CPU 图像预处理优化（2026-09-26）

这轮只减少重复的 PIL 像素复制，不改模型、检测阈值、裁剪几何、图库、身份阈值或匹配计算。完整开发原图的 275 条非计时输出与改动前一致；本机 CPU 耗时下降。它不是新的准确率实验，也不是飞腾派测速。

## 改动

原来读取图片、进入完整画面 `replay()`、裁剪后进入身份 `preprocess()`，都会执行 `ImageOps.exif_transpose(image).convert("RGB")`。即使图像已经转正且为 RGB，Pillow 也会复制像素。

`experiments/identity/runtime.py` 增加一个共享的 `oriented_rgb()`：

- 先解码，再读取 EXIF；需要旋转/镜像时仍使用 Pillow 原有变换。
- 仅在非 RGB 时转换颜色；已经转正的 RGB 图直接复用。
- helper 可能返回输入对象，调用方只读，不原地修改像素。
- 保留原有方形灰色填充、PIL bilinear resize、float32 归一化和 float64 图库匹配计算。

`read_image()` 在文件关闭前完成解码，仍支持转正原图坐标下的离线 `crop_box`。完整画面 CLI 也使用这个入口。不引入新的推理依赖或网络分支。

## 一致性验证

固定 SSDLite320 检测器、未微调 Large224 身份模型，以及 `crop-study/large224/fixed/detected.gallery.json`。仅使用原有 development 的 49 个身份 / 275 张原图，未读取 identity test 调参。

`preprocess_study.py` 逐张执行真实 `replay()`，保存检测框、检测分数、裁剪框、身份候选、身份分数、最终状态等全部非计时字段。优化版与改动前 **275/275 完全一致**，包括原有无框和多猫分支；错误预测也保持不变，并未借优化过滤困难样本。

新增两个测试覆盖：

- 8 种 EXIF 方向 × RGB/L/RGBA/CMYK/P，非方形图的输入张量逐元素等于旧变换，输入像素与方向不被修改。
- JPEG/PNG/TIFF 文件关闭后的像素访问、转正坐标裁剪，以及 RGB 快路径确实复用像素。

全仓库测试：`python -m pytest -q`，**90 passed**。这是实现回归，不是识别精度证明。

## 本机计时

Windows 11 x86-64，Python 3.12.10，NumPy 2.4.3，Pillow 12.3.0，ONNX Runtime 1.20.1，CPU 两线程。每个进程只有一个模型配置；缓存固定 20 张图，预热 20 次、计时 100 次。检测与身份模型均保持 FP32。

时间包括转正检查、检测预处理/网络/后处理、单框裁剪、身份预处理/网络与匹配；**排除文件读取/解码、模型加载和初始图片缓存**。不是摄像头到结果的延迟。

| 输入组 | 原复制路径 P50 / P95 | 优化后 P50 / P95 |
| --- | --- | --- |
| 身份开发集按 sample_id 排序前 20 张，3264×2448 / 4608×3072 原图；一组独立对照 | 166.44 / 234.72 ms | 115.79 / 168.12 ms |
| CatFLW 开发清单前 20 张，最长边 484–1024 像素；三个独立进程的范围 | 26.95–28.55 / 49.61–53.65 ms | 22.37–24.04 / 34.69–49.59 ms |

高分辨率组的 P50/P95 分别下降 30.43% / 28.37%。这是这组图片与本机配置下的一次对照，不承诺其他分辨率或设备也有相同比例收益。CatFLW 的 P95 波动明显，不挑选最快一轮宣称固定尾延迟收益。

CatFLW 三轮进程峰值 RSS 从 151.84–152.89 MiB 到 150.25–151.62 MiB，仍基本在同一范围；其中包含解码缓存和模型初始化，不能作为单帧端侧内存要求。高分辨率组缓存了 20 张大图，更不代表板端应同时保留这么多画面。

## 产物与复跑

完整仓库中的产物在 `experiments/runs/preprocess-study/`：

- `before/`：改源代码前保存的逐图结果，作为行为基准；主性能比较不用这次初始计时。
- `copy-reference/`：串行重跑旧复制行为，非计时输出再次与 `before/` 完全一致。
- `after/`：优化后结果，与 `copy-reference/` 精确比对。
- `before-catflw-{1,2,3}.json`、`after-catflw-{1,2,3}.json`：小图三轮独立进程计时。

旧复制行为只作为实验对照临时替换当前进程里的图像 helper，不进入运行时分支，也不保留另一套识别实现。后续无需恢复旧源码便可复跑：

```powershell
python -m experiments.detection.preprocess_study --reference-copies --output experiments/runs/preprocess-study/recheck-reference
python -m experiments.detection.preprocess_study --output experiments/runs/preprocess-study/recheck-optimized --compare experiments/runs/preprocess-study/recheck-reference
```

小图命令见 `experiments/detection/README.md` 的独立 CPU 测量，使用同一 CatFLW manifest、模型和 head 图库，每轮写不同输出文件。

优化后的代码更新进 `experiments/runs/catid-edge-detection-research.tar.gz`，检测/身份 ONNX 与研究图库不变。包仍只供研究重放；不含摄像头驱动、训练数据或门锁业务。

已将 21 文件研究包解到独立目录，用 `python -I` 从包内导入运行时代码，重放 6 张真实原图；known/unknown/unusable/no_cat 分支的非计时结果均与仓库版完全一致，进程没有导入 Torch。这个检查仍是在本机 x86 CPU 上完成，不是 ARM 安装或执行验证。

## 没有因此解决的问题

当前统一 head 裁剪的 Large224 在共同 39 组开发图库中的 DIR/FPIR 仍为 **82.87% / 2.41%**。本次没有提高准确率，也没有解决陌生猫拒识、暗光、实际拍摄场次或板端持续性能。继续保持研究候选，不进入 `src/catid/` 作为生产后端。
