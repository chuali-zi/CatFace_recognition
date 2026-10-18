# CatID · 猫个体识别

**接手项目先读：[项目交接与分工](docs/handoff-2026-10-18.md)。** 包含当前结论、数据/算法/板端三条工作线、运行入口、资产位置和验收边界。Git仓库包含源码、文档和脱敏板端报告；模型、照片、图库与离线运行时通过独立科研交接包提供。

**数据状态（2026-09-17）：已导出逐张目视检查的猫脸身份子集，1,369 张 / 247 个身份。** 使用 `data/clean.identity.jsonl`，可打开[清理后图库](data/review/cleaning/index.html)对照原图；详细处理与使用边界见[内容清理报告](docs/data-cleaning-2026-09-17.md)。`prepared.*` 是中间候选；9,743 条关键点候选尚未完成逐点语义审核。

这个仓库只负责猫咪身份识别：从图像判断是哪只已登记的猫，或返回陌生猫、没有猫、图像不可用。

**当前已有可运行的 MobileNetV3 Small/Large 猫脸身份实验，尚未选定生产推理模型。** 已打通真实图片基线、身份微调、ONNX 导出、陌生猫校准与 CPU 测量；仍需现场数据和飞腾派实测。实验不等于已完成完整画面识别服务。

3060 服务器与端侧操作入口：[轻量身份实验说明](experiments/identity/README.md)。本次实测与下一步取舍见[端侧准备报告](docs/edge-readiness-2026-09-22.md)。

飞腾派离线采集入口：[一键测试脚本](experiments/board/README.md)。离线包自带两种参照模型、冻结图库、149 张原图和可选 ARM64 依赖，板端执行 `bash run.sh`，输出识别指标、耗时、内存和设备信息。

2026-10-18 最新三份板端报告确认：修复后的包内离线安装、preflight 与 Small160/Large224 的 CPU 模型加载全部通过，尚未执行图片识别。v4完整科研测试包补齐149张冻结开发原图、两套研究图库和校准记录，执行 `bash run-all.sh` 一次完成两模型×1/2/4线程×三轮测量，回传一个结果ZIP。操作见[同学测试步骤](experiments/devicekit/QUICKSTART.md)，结论和边界见[板端报告分析](docs/board-test-2026-10-18.md)及[结果协议](docs/device-test-results.md)。

2026-09-22 已在本机 CPU 完成普通分类与 ArcFace 各 30 轮训练，并完成重复图库及 3/5 张登记图的严格配对对照。ArcFace 单组开发图库的优势没有稳定保持，见[图库稳定性报告](docs/gallery-study-2026-09-22.md)。后续四种主干/尺寸组合中，Large-224 提高了开发集平均正确接受率，但增加 CPU 成本且拒识未达标；Small-160 保留为速度参照。另完成 30 轮相似关系蒸馏，未取得净收益，结果见[本机主干与蒸馏对照](docs/backbone-study-2026-09-22.md)。

后续已完成 **SSDLite320 猫脸检测8轮CPU微调及ONNX导出**。在未按旧Haar命中筛选的CatFLW留出312图中，标注脸覆盖从Haar参照的185/312提升至302/312；人工核对63图为TP60/FP2/FN3。数据/指标边界和实测见[完整画面检测报告](docs/detection-study-2026-09-22.md)。

完整原图的身份重校准也已实跑：固定275张开发原图，273张产生单一候选；在共同39组有效图库中，Large224的正确接受率/陌生误认率为旧裁剪92.02%/2.23%、统一检测猫头裁剪82.87%/2.41%、固定脸部区域89.36%/2.61%。检测更好不等于身份更好；只换查询裁剪、保留旧登记图的结果更差。现已提供统一裁剪的原图登记与重放，默认猫头裁剪仍只是研究候选，见[裁剪与身份校准报告](docs/crop-study-2026-09-22.md)和[操作入口](experiments/detection/README.md)。

2026-09-26 又完成旧裁剪/新检测裁剪两组Small160各30轮CPU训练，严格使用相同145个训练身份、666张训练图和145张验证图。统一新裁剪训练优于错配训练，但完整原图开发DIR/FPIR为68.82%/2.86%，未超过不微调Small160的72.72%/1.34%；不替换研究包。完整对照及失败分组见[统一裁剪训练报告](docs/head-adaptation-2026-09-26.md)。

同日完成 CPU 预处理优化：去掉已转正 RGB 图的重复复制，275 张开发原图的检测与身份输出完全不变。高分辨率组本机 P50/P95 从 166.44/234.72 ms 降到 115.79/168.12 ms，较小图片另做三轮对照；均排除读取解码，不代表飞腾派速度。代码已更新到研究重放包，模型和阈值不变，见[预处理优化报告](docs/preprocess-optimization-2026-09-26.md)。

后续增加仅从145个训练身份/666图拟合的轻量特征变换对照。固定128维监督白化的开发DIR提高到90.25%，但FPIR升到2.87%，未形成同时改善，研究包仍不替换。已定位150个未进入现有协议的发布者身份、2,465张待复核原图，供下一步新身份验证；方法、负面结果与数据边界见[线性特征适配报告](docs/metric-study-2026-09-26.md)。

已继续完成新标签池的320张原图视觉审核，隔离1组跨旧标签身份疑点后，用39组/312张运行固定四臂对照。相同10组五猫图库中，Large224原特征DIR/FPIR为73.60%/2.28%，白化为67.60%/5.59%，上一轮收益没有保持；Small160为59.60%/1.40%。困难输入和检测失败保留分母，不替换默认模型，也不把该研究池自动并入clean。分母、重复框发现及限制见[新身份完整原图对照](docs/fresh-identity-study-2026-09-26.md)。

## 从哪里开始

1. [模块边界与交接接口](docs/scope.md)：了解识别模块提供什么。
2. [识别路线与实验安排](docs/research.md)：了解哪些路线已放弃、后续需要比较什么。
3. [数据说明](data/README.md)：了解本地数据的来源、标签与限制。
4. [评测协议](docs/evaluation.md)：了解校准、测试和指标的含义。

## 目录

```text
src/catid/
  contracts.py       识别结果与调用接口；尚无算法实现
  data.py            数据清单、统计、相同像素去重
  prepare.py         全量解码、跨源查重、关键点检查与任务子集
  evaluation.py      独立校准与开集身份指标
  __main__.py        统一命令行入口
tests/               上述工具的必要测试
docs/                模块边界、路线整理、评测协议
data/
  README.md          数据使用说明
  raw/               本地原始图片与原始标注，不进入版本控制
  samples.jsonl      统一清单，图片路径相对于 data/
  samples.reid-cleaned.jsonl  对 re-ID 去重后的派生清单
experiments/identity/ 轻量模型训练、开集评测、ONNX/INT8 与端侧打包
experiments/detection/ 猫脸检测训练、独立框评测、ONNX与完整画面实验重放
experiments/         数据内容清理与审计脚本
```

原始数据和生成清单是本地研究资产，复制项目到另一台机器时需要另外复制 `data/`。它们不随 Python 包分发。

## 安装与运行

在仓库根目录执行，建议使用虚拟环境：

```bash
python -m pip install -e ".[dev]"
python -m catid --help
python -m pytest -q
```

不安装包也可以运行，本机 PowerShell：

```powershell
$env:PYTHONPATH = "src"
python -m catid inspect data/samples.jsonl --out data/inventory.json
python -m catid clean data/samples.jsonl --sources reid --out data/samples.reid-cleaned.jsonl
python -m catid prepare data/samples.jsonl
python scripts/data_review.py data
```

Linux/macOS 使用 `export PYTHONPATH=src`，其余命令相同。安装后也可直接使用 `catid` 命令。

新候选模型输出逐样本预测后，统一评测：

```bash
python -m catid evaluate experiments/runs/my-run/predictions.jsonl --target-fpir 0.01 --out experiments/runs/my-run/metrics.json
```

预测文件需要由实际实验生成；输入格式见[评测协议](docs/evaluation.md)。仓库不附带冒充模型成绩的示例报告。

## 当前已验证的内容

- 新增 CatFLW 后共 39,860 条输入记录，全部图片成功解码、路径存在。
- 对 2,330 张猫脸候选逐张复核，输出 1,369 张 / 247 个身份，每身份 2–12 张，保留原图对应关系。
- 9,743 条关键点几何候选对应 7,946 张不同像素图像，逐点语义审核仍未完成，单独存放。
- 18,612 条待复核记录与 10,136 条精确重复记录保留清楚去向；四类输出互斥并覆盖全部输入。
- 身份 PNG 已逐张核对原图裁剪，精确重复和已记录排除规则漏入均为 0；原始图片保留。
- 数据清洗、陌生猫误认率、正确身份接受率、阈值并列分数、校准/测试隔离及命令行调用有自动测试。

新增的图片级身份实验已经可以生成真实模型成绩；实际相机效果和飞腾派性能仍待验证。发布者身份缺少现场真值和真实拍摄场次；后续评测需补充这些信息。

数据清理数字、图库入口和复跑方法见[内容清理结果](docs/data-cleaning-2026-09-17.md)。该报告记录的清理阶段未训练；2026-09-22 新增的模型实验另见上面的端侧准备报告。
