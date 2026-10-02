# 识别研究数据

**最新内容清理：身份入口为 `clean.identity.jsonl`，共 1,369 张猫脸 / 247 个身份。** 已逐张目视检查输出候选，核对原图裁剪，隔离已发现问题。见[内容清理报告](../docs/data-cleaning-2026-09-17.md)与[清理后图库](review/cleaning/index.html)。

| 最新文件 | 用途 |
| --- | --- |
| `clean.identity.jsonl` | 本次放行的猫脸身份子集，图片位于 `clean/faces/`，每身份 2–12 张 |
| `clean.landmarks.jsonl` | 9,743 条关键点几何候选，逐点语义审核仍为 pending，不是身份样本 |
| `clean.review.jsonl` | 18,612 条未放行原始记录和原因 |
| `clean.duplicates.jsonl` | 10,136 条精确重复记录，原图保留 |
| `clean.summary.json` / `clean.validation.json` | 实际统计和完整性检查 |
| `manual-group-review.csv` | 整个身份目录的持久排除规则，prepare 和内容导出均会应用 |

身份清单的 `path` 指向 EXIF 处理后、RGB 无损 PNG 猫脸；`original_path` 指向原始图片，`face_box` 为原图坐标。`review_status=agent_visual_checked` 表示模型代理目视检查，`identity_label_status` 说明保留发布者标签并做了组内视觉一致性检查，不是现场核实。`session_id` 仍为空；`leakage_group` 取整个身份。准备复制到 remote 时一起复制清单和相对路径指向的数据。

## 全量清理入口（2026-09-17）

新增 CatFLW 已下载并解包：2,079 张图、48 个关键点及脸框，没有猫身份标签。扩展清单 `samples.expanded.jsonl` 共 39,860 条。以下 `prepared.*` 是内容清理前的机械候选入口；当前实际身份放行名单见上面的 `clean.identity.jsonl`。

| 文件 | 用途 |
| --- | --- |
| `prepared.identity.jsonl` | 通过机械检查的身份研究子集，每标签至少两张；仍使用发布者标签，未完成人工身份审核 |
| `prepared.landmarks.jsonl` | 关键点格式、边界检查通过的子集；按 source 区分 CAT 9 点与 CatFLW 48 点 |
| `prepared.review.jsonl` | 近重复、异常图像、越界标注和 re-ID 来源问题复核队列 |
| `prepared.conflicts.jsonl` | 同像素跨身份或同方案标注冲突，整组隔离 |
| `prepared.duplicates.jsonl` | 精确重复排除记录，含 duplicate_of；原图不删除 |
| `prepared.singletons.jsonl` | 清理后不足两张的身份样本 |
| `prepared.unreadable.jsonl` | 无法解码记录和错误 |
| `prepared.near-pairs.jsonl` | 近重复候选对及距离，诊断记录而非样本清单 |
| `prepared.audit.jsonl` | 全量可解码图片检查记录 |
| `prepared.summary.json` | 实际输出数量和来源统计 |
| `review/index.html` | 本地分页原图复核入口 |
| `manual-review.csv` | 已目视确认的排除原因，重跑时自动应用到同像素副本 |

使用方法（未安装包时 PowerShell 先执行 `$env:PYTHONPATH = "src"`）：

```bash
python scripts/import_catflw.py data/downloads/catflw.zip
python -m catid prepare data/samples.expanded.jsonl
python scripts/data_review.py data
python -m catid inspect data/prepared.identity.jsonl
python -m catid inspect data/prepared.landmarks.jsonl
```

扩展清单已存在时不必重跑解包。清单和原图目录整体复制到 remote 后相对路径仍有效。prepared 中的原图需应用 EXIF 朝向并转 RGB；clean 身份图已经按这套处理保存为猫脸裁剪。

`prepare` 每 1,000 张保存 `prepared.scan.jsonl` 扫描进度；原始行内容、文件大小与修改时间均未变时复用缓存。强制重扫可删除这个派生缓存。重跑更新输入目录下同名派生文件，原清单不改写。

清理规则：

- 全来源完整解码，按应用 EXIF 后的尺寸和 RGB 像素查重。同身份重复留首张，跨身份冲突整组隔离。
- 全局 64 位 dHash 搜索原图和水平镜像，默认距离 ≤ 4 的候选进入复核，不合并身份。它不能穷尽裁剪、相邻帧或同猫不同图；相似背景可能误报。
- 短边不足 32 像素、近纯色图、关键点格式/朝向异常、越界点和非法脸框进入复核。越界坐标不强行裁回图内，保留 `landmark_in_frame` 掩码。
- 同图的 9 点与 48 点标注可分别保留；将来跨任务划分必须按 `pixel_hash` 一起分配，不能当作独立图。
- re-ID 抽查发现大量广告，部分 info.json 来自领养列表页；所有剩余 re-ID 图也进入复核，不因去重就视为可信身份数据。
- 抽查明确发现的多猫同框、目标不清图记录到 `manual-review.csv` 并进入复核；该记录不会把其他未检查图片自动标记为人工通过。
- 不改变人工 `review_status`，不伪造 `session_id`。人工身份与场次审核完成前不建立用于现场验收的最终划分；审核后的 clean 猫脸可单独建立明确标注为图片级诊断的训练/开发/测试划分，不能用它宣称事件级表现。

来源许可、新数据与教师模型入口见 [数据来源记录](../docs/data-sources.md)。CatFLW 原始文件位于 `raw/georgemartvel__catflw`，额外保留 `bounding_box`（x1,y1,x2,y2）和 `annotation_path`。

## 原有四个来源

以下数字来自新增 CatFLW 之前的本地清单，不等同于发布者宣称的规模。这些记录的人工审核状态仍为 `pending`，尚无可靠的拍摄场次标注。

| source | 原始目录 | 记录数 | 身份标签数 | 用途 |
| --- | --- | ---: | ---: | --- |
| `reid` | `raw/cronenberg64__cat-re-identification-image-dataset` | 1,880 | 250 | 身份研究；存在大量重复与跨身份冲突 |
| `individuals` | `raw/timost1234__cat-individuals` | 13,106 | 510 | 身份研究；仍需人工审核与近重复清理 |
| `hsc` | `raw/tobiastrein__heellostreetcat-individuals` | 2,801 | 68 | 固定场景研究；相邻帧、视角与场次需要隔离 |
| `catface` | `raw/crawford__cat-dataset` | 19,994 | 不适用 | 原始关键点标注；不是身份训练数据 |

原始图片及附属标注保留在 `raw/`。已删除重复下载压缩包、旧特征缓存、自动审核网页和旧模型生成的伪标签。使用或发布数据前应核对各原始数据集的使用条款；本次目录整理未重新审核许可。

## 清单格式

`samples.jsonl` 每行一个 JSON 对象：

```json
{"path":"raw/dataset/cat-a/image.jpg","source":"individuals","task":"identity","cat_id":"individuals:cat-a","session_id":null,"review_status":"pending","landmarks":null}
```

- `path`：相对于清单所在目录；不含本机盘符。移动整个 `data/` 后仍有效。
- `source`：数据来源。`cat_id` 带来源前缀，避免不同数据源同名目录直接合并。
- `task`：`identity` 或 `landmarks`。关键点样本的 `cat_id` 必须为 `null`，保留原始关键点坐标。
- `session_id`：同一次拍摄/来访的标识，未知时为 `null`。不能拿文件名随意生成假场次。
- `review_status`：原始和 prepared 清单保持 `pending`；本次 clean 身份输出为 `agent_visual_checked`。相同像素去重本身不会改变审核状态。
- `landmarks`：原图坐标的原始关键点；当前身份数据不附加自动预测关键点。

这次迁移从已有清单重建，只保留图片、来源、身份标签和原始关键点；旧自动框、启发式质量、异常分数与相似身份猜测全部去掉。清单不包含训练/验证/测试划分，划分必须在审核后明确建立。

## 旧版局部清理记录

```bash
python -m catid inspect data/samples.jsonl --out data/inventory.json
python -m catid clean data/samples.jsonl --sources reid --out data/samples.reid-cleaned.jsonl
python -m catid inspect data/samples.reid-cleaned.jsonl
```

`clean` 对指定身份数据源解码、应用 EXIF 朝向、转换 RGB 后比较相同像素。相同像素对应不同身份时整组排除；同身份重复保留清单中的第一张；无法读取的图片排除。原始图片不修改，其他数据源原样传入输出清单。

`samples.reid-cleaned.summary.json` 给出数量与逐图片排除原因。输出清单须与输入放在同一目录，以保持相对图片路径有效。也可一次选多个身份数据源：`--sources reid individuals hsc`；此次只实际重跑了 re-ID。

本次实测：re-ID 的 1,880 张图片中，834 张因跨身份相同像素冲突排除，190 张因同身份重复排除，保留 **856 张 / 215 个身份**；加上原样保留的其他来源，派生清单共 36,757 条，路径缺失为 0。

**re-ID 去重后清单仍包含未经清洗的其他来源。** 该清单不是可直接训练的审核通过集，不代表跨源身份一致性、近重复和错误身份已解决。CAT 的镜像/重复目录也尚未去重，不能把其 19,994 条记录解释为独立原始图片数。

重新下载的数据必须依据实际目录和标注生成清单；CatFLW 使用本次新增的专用导入脚本。新增样本后运行 `inspect` 和 `prepare`，并核对身份语义。
