# 完整画面猫脸检测实验

这是身份识别前端的研究候选，不是已验收的识别服务，不涉及门锁控制。
只用本机 CPU；训练/导出沿用 `../identity/requirements.txt`，Haar 另用已安装的 OpenCV。
ONNX 推理只需要 Pillow、NumPy 和 ONNX Runtime，不应加载 PyTorch。

## 数据与划分

CatFLW 发布者提供猫脸 xyxy 框，不是猫个体标签；不使用其 48 个关键点。
从 `data/prepared.audit.jsonl` 的全部 CatFLW 原图开始，不依赖 Haar 命中、旧猫脸清理结果或 `clean.identity`。
检查像素审计缓存的文件大小和修改时间及 EXIF=1，按同像素合并，再把直接/镜像 dHash 距离不超过 4 的连通组放在同一 split。
同像素重复标注框 IoU>=0.75 时保留路径排序靠前的框，避免把同一张脸重复计数。

固定 seed=20260922，70/15/15 按组切分：2,079 条原始记录 → 2,071 张不同像素图，2,067 组；train 1,447、development 312、test 312。
这不是官方划分、猫身份隔离或真实拍摄场次隔离；dHash 不能保证找全裁剪/转载近重复。COCO/ImageNet 初始化和 Haar 未完整公开的训练图片都不能保证与本地网络图逐图无交集。“独立”仅指没有按本仓库此前的 Haar 命中筛选。

```powershell
python -m experiments.detection.prepare --out experiments/runs/detection-cpu/data-v2
python -m experiments.detection.review --manifest experiments/runs/detection-cpu/data-v2/manifest.jsonl --out experiments/runs/detection-cpu/data-v2/reviewed.jsonl
```

输出目录必须为空；已有数据可直接复用，不要覆盖原图。
发布者通常只标一张脸，可能遗漏同框其他脸，所以全量默认 `annotation_scope=single_target`，只能统计已标注脸覆盖率，不能据多余预测框统计误报。

`review-decisions.json` 保存了独立于预测的完整画面目视复核：开发 64 张、test 64 张。
其中 127 张可确认各有一张可见猫脸；另 1 张存在镜面猫脸，保持非完整标注，不进入 exhaustive precision 分母。
`reviewed.jsonl` 只包含已复核的开发 64、test 63 张；保留模糊、侧向、旋转、遮挡和小脸，不按检测成败剔除。
这个子集只有正样本，precision 不等同无猫画面误报率，也不等同门口事件表现。

数据及派生权重只作为本仓库的研究实验使用。[CatFLW 作者仓库](https://github.com/martvelge/CatFLW)标明 CC BY-NC 4.0；不能把这批数据训练的研究包当成商业部署授权。

## Haar 固定对照

只比较 OpenCV basic/extended × 最长边 320/640 四组；固定 scaleFactor=1.1、minNeighbors=3、minSize=32、2 线程。
模型只初始化一次，逐图保存原图坐标预测；检测时间含 RGB resize、转灰度、扫描，不含图像解码。

```powershell
python -m experiments.detection.evaluate --manifest experiments/runs/detection-cpu/data-v2/manifest.jsonl --backend haar --cascade extended --max-side 320 --split development --out experiments/runs/detection-cpu/haar-extended-320-development
```

`summary.json` 给出 IoU>=0.5 的一对一目标匹配及尺寸分组；不是 AP。
不同框定义也会影响 IoU，不能把每个未匹配案例直接称为“完全没看到猫”。
开发评测发现 Haar 眼鼻框小于作者含耳朵框，因此额外固定一个 `--head-context` 对照：宽高扩大 1.5 倍、中心上移原框高度的 0.15 倍、裁到画面边界。只在 development 比较，不能把框定义适配当成新的网络能力。

## 轻量脸框微调

[官方 SSDLite320-MobileNetV3-Large](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html) 的 COCO_V1 仅用于初始化：它原来检测猫身，不能直接算猫脸模型。
本实验保留 91 类头，使用 class 17 学习作者猫脸框；训练后仅输出该类。冻结前段特征，仅训练最后特征块和检测头，所有 BatchNorm running statistics 冻结。
输入统一 PIL RGB bilinear 拉伸到 320×320、/255，再由模型内部归一化；随机水平翻转同步变换框。

```powershell
python -m experiments.detection.train train --manifest experiments/runs/detection-cpu/data-v2/manifest.jsonl --data-root data --out experiments/runs/detection-cpu/ssdlite320 --epochs 8 --batch-size 8 --threads 4
python -m experiments.detection.train predict --manifest experiments/runs/detection-cpu/data-v2/manifest.jsonl --data-root data --checkpoint experiments/runs/detection-cpu/ssdlite320/best.pt --split development --out experiments/runs/detection-cpu/ssdlite-development.jsonl
python -m experiments.detection.evaluate --manifest experiments/runs/detection-cpu/data-v2/manifest.jsonl --predictions experiments/runs/detection-cpu/ssdlite-development.jsonl --split development --out experiments/runs/detection-cpu/ssdlite-development
python -m experiments.detection.evaluate --manifest experiments/runs/detection-cpu/data-v2/reviewed.jsonl --predictions experiments/runs/detection-cpu/ssdlite-development.jsonl --split development --out experiments/runs/detection-cpu/ssdlite-reviewed-development
```

AdamW 1e-3，weight_decay=1e-4；只用 train 更新权重、development loss 选择 checkpoint，不自动运行 test。
训练默认 CPU；服务器可用后相同入口可显式传 `--device cuda`，本轮没有运行 CUDA。
原始 Torch predict 默认 score>=0.3，最终候选已只根据开发子集比较0.3/0.5/0.7后选择0.7；用Torch复跑最终配置应显式加 `--threshold 0.7`。最多10框、模型NMS=0.55，坐标还原至原图。
发布者标注不完整时，标准 SSD loss 可能把漏标猫当背景惩罚，这是当前训练的局限，不是已解决的问题。
测试保持按固定配置一次检查，不据 test 回调阈值。所有 CPU 时间必须说明本机环境，不能按倍数外推飞腾派。

新检测框与旧身份 crop 分布不同。即使检测覆盖率提高，也不能直接继承旧登记库的身份阈值或宣称端到端识别率提高；需要重新用实际检测 crop 做登记/校准/查询对照。

## ONNX 与完整画面重放

最终模型与运行结果见[检测实测报告](../../docs/detection-study-2026-09-22.md)。ONNX图内是SSDLite归一化，不要给它输入身份模型的ImageNet预处理tensor。

```powershell
python -m experiments.detection.onnx export --checkpoint experiments/runs/detection-cpu/ssdlite320/best.pt --out experiments/runs/detection-cpu/onnx/model.onnx --manifest experiments/runs/detection-cpu/data-v2/manifest.jsonl --data-root data --score-threshold 0.7
python -m experiments.detection.onnx predict --model experiments/runs/detection-cpu/onnx/model.onnx --manifest experiments/runs/detection-cpu/data-v2/manifest.jsonl --data-root data --split test --out experiments/runs/detection-cpu/ssdlite-test.jsonl --threads 2
```

导出时对开发前5张做官方Torch完整输出与ONNX+NumPy对照，数量/框/分数不一致会报错。模型旁必须保留 `model.onnx.json`，含已固定阈值、NMS和训练来源。

```powershell
python -m experiments.detection.replay --image data/raw/timost1234__cat-individuals/cat_individuals_dataset/0004/0004_048.JPG --detector-model experiments/runs/detection-cpu/onnx/model.onnx --identity-model experiments/runs/backbone-study/large224/model.onnx --gallery experiments/runs/crop-study/large224/fixed/detected.gallery.json --threads 2
```

这是已在开发集上用统一检测crop重新登记/校准的研究图库，不是旧Haar图库。完整对照见[裁剪报告](../../docs/crop-study-2026-09-22.md)。结果仍为 `diagnostic_only`；无检测框和多候选时不运行身份模型。检测权重或旁侧metadata（含score/NMS）与校准不一致会报错，裁剪方式从图库读取。旧图库仍可做明确标为未端到端校准的诊断，不继承旧识别成绩。

## 原图裁剪与身份对照

冻结检测器与身份模型，使用既有protocol的development原图，不读取identity test。旧face PNG只作参照，原参考框仅用于诊断IoU，不输入新检测或挑选框。

```powershell
python -m experiments.detection.crops --detector experiments/runs/detection-cpu/onnx/model.onnx --models small160=experiments/runs/edge160/baseline/model.onnx large224=experiments/runs/backbone-study/large224/model.onnx --protocol experiments/runs/edge160/protocol.json --crop-mode head --out experiments/runs/crop-study/cache --threads 2
python -m experiments.detection.crop_study --cache-dir experiments/runs/crop-study/cache --protocol experiments/runs/edge160/protocol.json --model-name large224 --out experiments/runs/crop-study/large224
```

cache目录需为空；已有结果可直接读。改为 `--crop-mode face_core` 和新cache/out路径可复现唯一几何对照：检测框宽高取2/3，中心下移框高0.1。它没有同时改善正确接受和陌生误认，因此默认仍为head。study的三mode分别为legacy、mixed（旧登记、新calibration/query，仍重新校准）、detected；登记失败不补图，query失败保留分母。

## 从自己的完整图片登记

清单每行一个JSON，路径相对于清单文件；`cat_id`必须真实区分猫，gallery和calibration_unknown的身份不能重叠。以下只是格式示例，不包含可用于成绩的样本：

```jsonl
{"path":"images/my-cat-01.jpg","cat_id":"my-cat","split":"gallery"}
{"path":"images/other-cat-01.jpg","cat_id":"other-cat","split":"calibration_unknown"}
```

应提供多张不同姿态的登记图及多只陌生猫的校准图；这里不规定几张就能保证效果，也不能把训练集、同一帧副本或已登记猫用于陌生校准。研究只覆盖公开图，目标1%是校准设置，不是实际达到的保证。

```powershell
python -m experiments.detection.enroll --manifest my-enrollment.jsonl --detector experiments/runs/detection-cpu/onnx/model.onnx --model experiments/runs/backbone-study/large224/model.onnx --crop-mode head --threads 2 --out my-gallery.json
```

采用与replay同一EXIF/RGB处理、单框规则和裁剪函数。路径/像素重复、身份角色冲突和无法解码的输入会报错；计划gallery中任何图片无框/多框会拒绝生成图库并写 `.failures.json`。calibration中无框/多框会记录并排除，全部不可用则报错。输出记录计划/成功校准数，不覆盖已有图库。重放时使用 `--gallery my-gallery.json`，不要修改JSON中的crop_mode冒充重校准。

本机已从126张完整原图（15登记+111陌生校准）实际重建head/Large224图库：15登记成功，109校准成功，图库向量和阈值与cache研究结果完全一致。这验证了实现一致性，不是新的识别准确率。

## 无Torch的研究重放包

```powershell
python -m experiments.identity.package --mode edge --model-dir experiments/runs/backbone-study/large224 --gallery experiments/runs/crop-study/large224/fixed/detected.gallery.json --detector-dir experiments/runs/detection-cpu/onnx --out experiments/runs/catid-edge-detection-research.tar.gz
```

解包后安装 `experiments/identity/requirements-runtime.txt`，给一张本地真实图片即可运行，无需摄像头：

```bash
python -m experiments.detection.replay --image /path/to/frame.jpg --detector-model detector/model.onnx --identity-model model/model.onnx --gallery model/gallery.json --threads 2
python -m experiments.detection.enroll --manifest /path/to/my-enrollment.jsonl --detector detector/model.onnx --model model/model.onnx --crop-mode head --out my-gallery.json --threads 2
```

包含检测与身份两个ONNX、对应metadata、重新校准的head研究图库、原图登记/重放代码及最小阈值工具；不含Torch、训练数据或摄像头驱动。附带图库中的5只猫只是公开图研究身份，不是用户自己的猫。已有 `catid-server.tar.gz` 是上一阶段身份数据包，不含这次CatFLW检测训练数据。

## 独立CPU测量

```powershell
python -m experiments.detection.benchmark --manifest experiments/runs/detection-cpu/data-v2/manifest.jsonl --backend pipeline --model experiments/runs/detection-cpu/onnx/model.onnx --identity-model experiments/runs/backbone-study/large224/model.onnx --gallery experiments/runs/crop-study/large224/fixed/detected.gallery.json --threads 2 --warmup 20 --runs 100 --out experiments/runs/crop-study/benchmark-pipeline.json
```

`--backend haar --cascade extended --max-side 320 --head-context` 测固定Haar；`--backend onnx --model ...` 只测SSDLite。
每次只加载一个配置，缓存相同20张开发图片；时间排除文件解码和模型加载。pipeline另报实际运行身份的次数，跳过分支不混入身份条件耗时。飞腾派仍需用板端真实图片、完整解码与持续运行另测。

2026-09-26 已减少读取/重放/特征预处理之间的重复 RGB 图像复制。275 张原图结果与修改前完全一致，权重与图库不变；研究包已带上此优化。高分辨率组和较小图片的耗时分别记录，见[预处理优化报告](../../docs/preprocess-optimization-2026-09-26.md)。在完整研究仓库可用 `python -m experiments.detection.preprocess_study --help` 复跑对照；该评测脚本和数据不放入端侧包。
