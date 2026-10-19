# 轻量猫脸身份实验与端侧准备

可选的候选歧义拒识诊断见 [AMBIGUITY_STUDY.md](AMBIGUITY_STUDY.md)。默认关闭，保留原余弦阈值；用于已曝光开发分数对照或实际单图诊断，不直接修改端侧默认规则。

本目录提供真实图像的特征基线、普通分类/ArcFace 身份微调、ONNX 导出、开集评测、静态 INT8 对照和 CPU 测量。默认轻量参照是 **MobileNetV3-Small，160×160，576 维单位向量**，使用分类器之前的池化特征。后续本机对照新增 **Large-224，960 维**精度候选，尚未选定生产识别模块。输入默认是已经裁好的猫脸，完整画面的检测另见 `detect.py` 与 [检测准备](../../docs/detection-readiness.md)。

统一检测crop的同样本训练对照见[2026-09-26实验说明](../../docs/head-adaptation-2026-09-26.md)。训练/导出支持行内可选整数 `crop_box`：先EXIF转正，再按原图xyxy裁剪；检测离线完成，不在每个训练轮次重复运行。完整原图的登记、重校准与回放入口见[检测实验](../detection/README.md)。

新发布者标签的完整原图审核及固定四臂CPU对照见[新身份实验](../../docs/fresh-identity-study-2026-09-26.md)。`fresh_data` 生成待审核候选并合并逐图决定，`fresh_study` 只使用冻结角色、检测框和旧训练集变换；不会重新拟合变换或覆盖已有结果。本轮白化不优于原特征，不更新默认研究包。

## 安装

从仓库根目录执行，Python 3.10 及以上，建议独立虚拟环境：

```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

3060 服务器先按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 安装与驱动匹配的 CUDA 版 `torch` / `torchvision`，再安装实验依赖。不要将本机 Windows 的 CPU wheel 复制到服务器。

```bash
python -m pip install -r experiments/identity/requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
python -m pytest -q
```

`--device cuda` 在 CUDA 不可用时会明确失败。尚未连接 3060 时可指定 `--device cpu` 验证全流程。板端只需 `requirements-runtime.txt`，不装 PyTorch、训练器或 ONNX 转换工具。

## 一次完成服务器开发实验

将 `data/clean.identity.jsonl` 和所引用的 `data/clean/faces/` 一起复制，约 1.01 GiB 原始 PNG；不需要复制所有原图来训练身份候选。

```bash
python -m experiments.identity.workflow --out experiments/runs/server160 --device cuda --epochs 30 --batch-size 32 --workers 0 --threads 2 --cache-images --loss arcface
```

此命令依次生成固定划分、跑无微调基线、训练、导出最优训练身份验证 loss 的 checkpoint、比较开发集，并可生成 INT8 开发对照。它不会读取最终测试查询来选模型。输出目录必须是新的，失败的独立阶段可用下面命令重跑。默认不启用 AMP，3060 可在单独 `train` 命令加 `--amp`。

`--cache-images` 将一次解码和缩放的训练/验证 tensor 缓存在内存，本数据约 242 MiB，避免每轮重新读取约 1 GiB PNG；随机翻转仍每次采样执行。不缓存时可用 `--workers 4` 并行解码。需要研究量化时显式加 `--int8`；本轮已观察到的 INT8 精度损失见下文。

另可加 `--loss arcface` 对照身份训练目标（默认角度 margin 0.3、scale 30）。[ArcFace 原论文](https://arxiv.org/abs/1801.07698) 的归一化特征/类别权重与目标角度间隔只作用于训练分类头，导出的 576 维特征网络不增加层数。这里是猫脸数据的新实验，不继承人脸论文的识别率；验证 loss 使用带 margin 的分数，分类 accuracy 使用不带 margin 的归一化余弦分数。两种损失均用相同训练身份、增强、初始化和 30 轮预算。

本机固定协议默认种子 20260922，5 只登记猫、每猫 3 张注册图。148 个训练身份（678 张训练、148 张验证），49 个开发身份和 50 个最终测试身份。每个评测分区的图库猫、阈值校准陌生猫、测试陌生猫互不重叠。训练/开发/最终测试身份也互不重叠。已登记猫注册与查询按图片分离；**没有真实场次，不能证明跨来访表现**。公开预训练 ImageNet 与猫图来源的潜在重合尚未审计。

主干来源是 [Torchvision 官方 MobileNetV3-Small](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html)，明确使用 `IMAGENET1K_V1`。本实验为完整保留已裁猫脸，采用 EXIF 转正、RGB、128 灰色居中补方（偏移向下取整）、PIL bilinear 缩为 160、除 255、ImageNet mean/std；这与官方 ImageNet 的 224 中心裁剪不同。训练只增加水平翻转，推理不增强。每次导出用 5 张真实训练图检验 PyTorch/ONNX 输出一致。

## 分阶段复跑

```bash
python -m experiments.identity.protocol --manifest data/clean.identity.jsonl --out experiments/runs/my160/protocol.json --gallery-size 5 --enrollment 3
python -m experiments.identity.run export --protocol experiments/runs/my160/protocol.json --out experiments/runs/my160/baseline
python -m experiments.identity.run evaluate --model experiments/runs/my160/baseline/model.onnx --protocol experiments/runs/my160/protocol.json --out experiments/runs/my160/baseline/development
python -m experiments.identity.train --manifest data/clean.identity.jsonl --protocol experiments/runs/my160/protocol.json --out experiments/runs/my160/trained --device cuda --epochs 30 --batch-size 32 --workers 4 --threads 2
python -m experiments.identity.run export --checkpoint experiments/runs/my160/trained/best.pt --protocol experiments/runs/my160/protocol.json --out experiments/runs/my160/trained
python -m experiments.identity.run evaluate --model experiments/runs/my160/trained/model.onnx --protocol experiments/runs/my160/protocol.json --out experiments/runs/my160/trained/development
```

固定划分上的 ArcFace 对照：

```bash
python -m experiments.identity.train --manifest data/clean.identity.jsonl --protocol experiments/runs/my160/protocol.json --out experiments/runs/my160/arcface --device cuda --epochs 30 --threads 2 --cache-images --loss arcface --margin 0.3 --scale 30
python -m experiments.identity.run export --checkpoint experiments/runs/my160/arcface/best.pt --protocol experiments/runs/my160/protocol.json --out experiments/runs/my160/arcface
python -m experiments.identity.run evaluate --model experiments/runs/my160/arcface/model.onnx --protocol experiments/runs/my160/protocol.json --out experiments/runs/my160/arcface/development
```

`metrics.json` 包含 Top-1、正确身份接受率 DIR、陌生猫误认率 FPIR、已知猫错认/拒识和分母；`predictions.jsonl` 可用已有 `catid evaluate` 再算。`gallery.json` 保存实际评测的平均原型和阈值，绑定模型文件，换模型或更改登记图后必须重新生成与校准。它是离线实验图库，不是已审核的家庭猫登记库。

开发对照结束后，确定候选再执行最终测试；本轮不要继续用测试成绩调整超参数：

```bash
python -m experiments.identity.run evaluate --model experiments/runs/my160/trained/model.onnx --protocol experiments/runs/my160/protocol.json --partition test --out experiments/runs/my160/trained/test
```

当前陌生校准只有约百张图片、来自约二十只猫；1% 是经验校准目标，有限相关样本不能支持现场总体 1% 误认保证。5 猫图库的已知查询也很少，报告分子分母，不仅百分比。研究已训练模型的其他图库时，应使用下面的 `study` 在原有开发身份池内重抽，不能重新打散完整数据的训练/开发身份。

## 图库与登记数量稳定性

`study` 已实跑基线、普通分类、ArcFace 的四种匹配方案：平均原型或逐张最大，使用余弦或第一二身份分差。默认仅研究原协议开发身份，2/5 猫、3/5 张登记图、10 个固定种子；每组图库/方法独立用陌生校准图选阈值。`--paired-enrollment` 保持图库身份及全部查询不变，仅增加两张登记图。后者的 3 张方案不把额外两张图挪到查询集。

```bash
python -m experiments.identity.study --protocol experiments/runs/my160/protocol.json --models baseline=experiments/runs/my160/baseline/model.onnx trained=experiments/runs/my160/trained/model.onnx --out experiments/runs/my160/gallery-study --threads 2
python -m experiments.identity.study --protocol experiments/runs/my160/protocol.json --models baseline=experiments/runs/my160/baseline/model.onnx trained=experiments/runs/my160/trained/model.onnx --paired-enrollment --cache-dir experiments/runs/my160/gallery-study/cache --out experiments/runs/my160/paired-study --threads 2
```

输出包含固定 `suite.json`、每种方案的逐 episode 预测和 macro 汇总；已有特征缓存可复用。重复图片不能累加为新增独立样本，图库规模之间也会改变陌生猫组成。默认非配对模式不能单独证明多两张登记照片的因果收益。真实结果与边界见[图库稳定性报告](../../docs/gallery-study-2026-09-22.md)。这些研究输出不生成可部署的新匹配算法图库，现有端侧入口仍仅支持平均原型/余弦。

## 主干与输入尺寸对照

未微调 Small/Large、160/224 的四组合已在同一开发图库协议和本机 CPU 上实跑，详见[主干对照报告](../../docs/backbone-study-2026-09-22.md)。Large-224 的平均正确接受较高，但拒识仍未达标，且本机特征开销约为 Small-160 的 3 倍。Small-160 保留为速度参照，Large-224 作为后续精度候选；不是无条件替换关系。

```bash
python -m experiments.identity.run export --architecture mobilenet_v3_large --size 224 --protocol experiments/runs/my160/protocol.json --out experiments/runs/my160/large224 --threads 2
python -m experiments.identity.run evaluate --model experiments/runs/my160/large224/model.onnx --protocol experiments/runs/my160/protocol.json --out experiments/runs/my160/large224/development --threads 2
```

明确使用官方 `IMAGENET1K_V1` 权重；Large 首次导出会下载约 21 MiB 的官方 checkpoint，导出时移除分类器，ONNX 约 11.33 MiB。Large 当前只支持未微调导出，不能给旧 Small 的训练 checkpoint 加 `--architecture mobilenet_v3_large`。图库/运行时按模型 metadata 校验 576/960 维度，换模型或尺寸必须重建图库和阈值。

另有独立的 `distill` 研究入口：将冻结 ONNX 教师的批内余弦关系教给 Small-160，无分类头或推理投影层。教师原图/翻转特征与学生翻转同步；只用原协议训练身份，按训练身份留出图的完整 pairwise loss 选 checkpoint。当前 30 轮实际结果未取得净收益，命令用于复现实验，不是推荐训练配方。

```bash
python -m experiments.identity.distill --teacher experiments/runs/my160/large224/model.onnx --manifest data/clean.identity.jsonl --protocol experiments/runs/my160/protocol.json --out experiments/runs/my160/distilled160 --epochs 30 --threads 4 --device cpu
python -m experiments.identity.run export --checkpoint experiments/runs/my160/distilled160/best.pt --protocol experiments/runs/my160/protocol.json --out experiments/runs/my160/distilled160 --threads 2
```

`distill` 拒绝覆盖非空训练输出目录。默认学生输入 160，导出后仍可使用 `run evaluate`、`study` 对照；不要把较低蒸馏 loss 或单组零误认当作跨图库识别改善。

## 量化与飞腾派测量

静态量化仅使用训练分区的 100 张真实图片估计激活范围，与陌生猫阈值校准是两件事。使用 [ONNX Runtime 官方 QDQ 静态量化](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)。INT8 必须独立重建图库、重新校准并评测；不要因文件更小就默认使用。

2026-09-22 的 ORT 1.20.1 本机实验中，基线 S8S8 INT8 的开发 DIR 从 22/29 降至 10/29；U8U8 和末层保留浮点的有界对照也未恢复特征一致性。当前优先 FP32，下面的量化命令保留用于研究复跑，不是推荐部署参数。量化流程中的范围/偏置警告应结合输出和真实身份评测检查，不能只看模型是否成功加载。

```bash
python -m experiments.identity.deploy quantize --model experiments/runs/my160/trained/model.onnx --protocol experiments/runs/my160/protocol.json --out experiments/runs/my160/int8/model.onnx
python -m experiments.identity.run evaluate --model experiments/runs/my160/int8/model.onnx --protocol experiments/runs/my160/protocol.json --out experiments/runs/my160/int8/development
python -m experiments.identity.deploy benchmark --model experiments/runs/my160/trained/model.onnx --manifest data/clean.identity.jsonl --gallery experiments/runs/my160/trained/development/gallery.json --threads 2 --out experiments/runs/my160/trained/benchmark-t2.json
```

在实际飞腾派分别测 `--threads 1`、`2`、`4`。报告模型加载、首次推理、预热后 P50/P95、解码预处理、特征、图库匹配和进程峰值内存。`face_crop_total` 不含检测或相机，不能称端到端门口延迟。重复文件有系统缓存收益，板端还需持续运行温度和降频观察。本机 x86 数字不能按倍数外推。暂定开发取舍参考：身份特征 P95 尽量低于 100 ms、整图流程后续争取 500 ms 内得到结果；这些是待板端验证的工程目标，不是已达标成绩。

## 打包与图片重放

自己的猫登记图准备好后，在服务器或完整仓库运行 `enroll`。登记清单路径相对于该 JSONL 所在目录，格式如下（仅为格式示例）：

```jsonl
{"path":"faces/home-a-01.png","cat_id":"home-a","split":"gallery"}
{"path":"faces/stranger-b-01.png","cat_id":"stranger-b","split":"calibration_unknown"}
```

实际应有多张登记图和足够多的独立陌生猫，身份与拍摄场次安排见现场协议。工具检查身份隔离、路径与相同像素重复；没有陌生校准图就拒绝生成阈值。

```bash
python -m experiments.identity.enroll --manifest /path/to/enrollment.jsonl --model experiments/runs/my160/trained/model.onnx --out experiments/runs/my160/home-gallery.json --target-fpir 0.01
```

新图库的阈值只适用于这份模型和登记图；还需要另外的已知/陌生查询评测，不能把校准目标当作验收成绩。该登记入口不依赖 PyTorch，但调用基础包的统一阈值函数，因此在完整仓库/服务器运行；只需将生成的 `gallery.json` 带到板端。

服务器打包包括代码、文档、审核猫脸和清单，不含 `.env`、`.git` 或全部原图：

```bash
python -m experiments.identity.package --mode server --out experiments/runs/catid-server.tar.gz
python -m experiments.identity.package --mode edge --model-dir experiments/runs/my160/trained --gallery experiments/runs/my160/trained/development/gallery.json --out experiments/runs/catid-edge.tar.gz
```

服务器可另外带 `--pretrained-weight <torch缓存中的mobilenet_v3_small-047dcff4.pth>`。解包后将 `weights/` 中的文件复制到 `torch.hub.get_dir()/checkpoints/`；随后官方模型构造器可复用缓存。服务器训练仍需正确安装 CUDA 版 PyTorch。

板端解包后在解包根目录执行（Linux ARM64，运行时 wheel 是否可用取决于实际 Python/系统）：

```bash
python -m pip install -r experiments/identity/requirements-runtime.txt
python -m experiments.identity.deploy recognize --model model/model.onnx --gallery model/gallery.json --image /path/to/cat-face.png
python -m experiments.identity.deploy benchmark --model model/model.onnx --gallery model/gallery.json --images /path/to/face1.png /path/to/face2.png --threads 2 --out board-t2.json
```

只提供裁好猫脸时，`recognize` 仅判断 known/unknown；它不会把风景/人脸可靠识别为 no_cat。`src/catid/contracts.py` 的完整四状态后端仍需独立验证的猫检测与质量判断。

完整图片的 Haar 实验重放可安装 `requirements-detection.txt` 后使用 `detect replay`；该候选用于发现流程问题，不作为生产检测器验收。

```bash
python -m experiments.identity.detect replay --image /path/to/full-frame.jpg --model model/model.onnx --gallery model/gallery.json --max-side 640 --threads 2
python -m experiments.identity.detect check --manifest data/clean.identity.jsonl --protocol experiments/runs/my160/protocol.json --out experiments/runs/my160/detection-check.json --limit 30 --threads 2
```

完整图 `check` 需要清单中的原始图片，默认身份服务器包仅带猫脸裁剪，因此须在保有原图的本机运行或额外复制涉及的原图。它只检查先前已被 Haar 筛选的正样本，不能称真实检测召回。`replay` 的检测裁剪也尚未做独立阈值校准。摄像头到位后的数据安排见[现场采集协议](../../docs/camera-validation.md)。
