# 猫脸检测裁剪与身份特征对照（2026-09-22）

本报告是开发集上的图像级诊断，不是生产能力或身份验收。研究比较旧的人工审核 face crop（`legacy`）、旧 crop 登记而检测 crop 用于 calibration/query 的 `mixed`，以及登记、calibration、known、unknown 全部使用检测 crop 的 `detected`。检测 crop 使用本轮唯一固定的 SSDLite320 猫脸检测候选；它不是逐张参考框拟合，也不是把参考框当作推理输入。

## 数据和协议边界

开发池共 275 帧：273 帧检测成功并有一个可用检测框，1 帧为 `unusable`，1 帧为 `no_cat`。失败登记不能从图库中静默删除；若计划登记图不可用，整个 episode 标记为 `enrollment_incomplete`，不拿 query 补登记，也不把失败写成正确拒识或 FPIR=0。calibration 只使用成功检测的陌生猫。

评测复用 `edge160/protocol.json` 的 development 分区和既有 `build_paired_suite`：2/5 个 gallery identity × 3/5 张登记图 × 10 个 seed。三种 mode 的共同有效交集为 39 个 episode；detected mode 另外有 1 个因登记失败而不完整的 episode：seed8、5猫、每猫5图，其 `0402_006` 为多猫图。重复图片来自同一固定身份池，不是独立猫、独立访问或独立事件，以下 macro 平均不能当作统计独立试验。

所有阈值均从每个 episode 的成功 calibration unknown 按 target FPIR=0.01 独立校准。DIR/FPIR 保留计划 query 作为全量分母；conditional DIR/FPIR 只在可用 query 上计算。共同39组的known query没有检测失败，conditional DIR等于DIR；unknown query存在失败，平均失败占比0.7874%，所以conditional FPIR略高。原始固定protocol的两张失败图恰好都在calibration，mixed/detected校准可用109/111，known29和unknown120全部可用。不能把固定protocol分母套到重复episode。

## small160：共同 39 episode macro

| mode | DIR | FPIR | conditional DIR | conditional FPIR |
|---|---:|---:|---:|---:|
| legacy | 0.8044 | 0.0314 | 0.8044 | 0.0314 |
| mixed | 0.3924 | 0.0346 | 0.3924 | 0.0348 |
| detected | 0.7272 | 0.0134 | 0.7272 | 0.0135 |

这里的 mixed 说明只把登记和查询裁剪语义混合会明显改变特征分布；detected 相比 mixed 恢复了部分 DIR，且共同交集上的 FPIR 较低，但仍未达到把它视为现场身份识别保证的程度。

## large224：共同 39 episode macro

| mode | DIR | FPIR | conditional DIR | conditional FPIR |
|---|---:|---:|---:|---:|
| legacy | 0.9202 | 0.0223 | 0.9202 | 0.0223 |
| mixed | 0.4929 | 0.0304 | 0.4929 | 0.0306 |
| detected | 0.8287 | 0.0241 | 0.8287 | 0.0243 |

large224 的 legacy 在这一开发诊断中高于 small160；但切换到统一 detected crop 后下降，仍高于 small160 detected。mixed 的明显下降再次表明登记与查询裁剪不一致是独立的分布变化，不能把 mixed 数字解释成检测器本身的纯贡献。

## 原始固定 development：5 猫、3 张登记图

固定 protocol 的 gallery 为 5 个 identity、每猫 3 张登记；known 29 帧、calibration unknown 111 帧、unknown 120 帧。下表是 fixed 评测，不是 39 episode macro：

| model | mode | calibration 可用 | DIR | known misidentification | identity reject | FPIR |
|---|---|---:|---:|---:|---:|---:|
| small160 | legacy | 111/111 | 0.7586 | 0.0690 | 0.1724 | 0.0333 |
| small160 | mixed | 109/111 | 0.3793 | 0.0345 | 0.5862 | 0.0000 |
| small160 | detected | 109/111 | 0.6552 | 0.0000 | 0.3448 | 0.0000 |
| large224 | legacy | 111/111 | 0.9310 | 0.0690 | 0.0000 | 0.0250 |
| large224 | mixed | 109/111 | 0.4138 | 0.0000 | 0.5862 | 0.0000 |
| large224 | detected | 109/111 | 0.8276 | 0.0345 | 0.1379 | 0.0000 |

mixed/detected 的 109/111 calibration 可用是失败检测按规则排除 calibration 校准样本，不是把失败样本设成低分或拒识。固定结果中没有 identity test、ARM/飞腾派实测或 event-level/连续帧验证。

## 解读限制和下一步

原图来自已经经过 Haar 参照筛选的身份池；因此本研究不能估计未筛选全画面的检测覆盖，也不能把 0 像素或 dHash 近重叠审计等同于身份独立。数据中的重复图片、同一身份多帧和固定 development 协议都限制了外推。

本轮只追加下面这一 `face_core` 几何candidate，在同一cache/39 episode设计下比较，没有增加其它检测候选或逐图调框。

## face_core 对照结果（同一 39 个共同有效 episode）

face_core 是看到head结果后、运行本候选前固定的唯一几何对照：从检测head框取精确2/3宽高，中心向下移动 `head_h*0.1`，最后一次round到原图像素。它对应此前Haar扩头框规则的逆几何，不是逐张参考框拟合，也不是搜索多个比例后挑出的最佳框。

| model | crop mode | DIR | FPIR | conditional DIR | conditional FPIR |
|---|---|---:|---:|---:|---:|
| small160 | head / detected | 0.7272 | 0.0134 | 0.7272 | 0.0135 |
| small160 | face_core / detected | 0.7406 | 0.0280 | 0.7406 | 0.0281 |
| large224 | head / detected | 0.8287 | 0.0241 | 0.8287 | 0.0243 |
| large224 | face_core / detected | 0.8936 | 0.0261 | 0.8936 | 0.0263 |

宏平均不能单独决定采用哪种裁剪。face_core 在这 39 个 episode 上提高了两种主干的 DIR，但 FPIR 同时上升，尤其 small160 从约 1.34% 上升到约 2.80%；因此不能仅挑 DIR 宏平均宣布改善。

固定 development（5 猫、3 张登记图）的结果也不支持直接改默认：large224 detected 从 head 的 24/29（0.8276）变为 face_core 的 21/29（0.7241），尽管陌生 unknown 在 face_core 固定结果中为 0/120 接受；small160 face_core 为 22/29 DIR，3 个 known 错认，陌生猫为 8/120 接受（FPIR 0.0667）。这些固定结果与 episode 宏平均方向并不完全一致，说明应保留逐设置和固定 protocol 两种观察。

Large224分设置的head→face_core同组对照也显示这种取舍：2猫3图DIR为79.94%→90.35%，FPIR为2.36%→3.74%；5猫3图则为81.43%→84.50%、3.18%→1.92%。不同gallery大小的陌生池也不同，不能把这里解释成单独增加猫数的因果效果。

本轮结论：head crop继续作为当前研究包默认对照；face_core仅保留为一个有界几何对照，不提升为生产默认，也不宣称已经解决拒识。本轮没有再运行identity test，没有ARM/飞腾派或event-level验证。

## 原图登记一致性与可复跑产物

通过新 `experiments.detection.enroll` 入口从完整原图独立运行检测/裁剪/身份推理，而非读取研究特征cache：15张gallery全部成功、111张calibration中109张成功。Large224/head的5个原型向量与study fixed图库逐元素最大差为0，阈值同为 `0.8642034895562595`，差为0。无框与多框均记录，没有被写成陌生低分。

`0402_006.JPG` 的多框经目视核实确有前景灰猫和右上橘猫；`0450_014.JPG` 是真实侧脸灰白猫漏检，`no_cat`指没有检测到，不是声称画面没有猫。另外核对3张低参考IoU图，主要是新含耳朵猫头框与旧眼鼻face框覆盖范围不同；IoU诊断不用于删图。

本轮数据及预测均保存在 `experiments/runs/crop-study/`：

- `cache/`、`face-core-cache/`：275原图的逐图检测、特征、来源与配置。
- `small160/`、`large224/`、`face-core-small160/`、`face-core-large224/`：三mode逐episode预测、共同交集摘要、四种图库设置摘要、固定protocol与图库。
- `enrollment-manifest.jsonl`、`enrolled-large224-head.json`：真实原图登记复核输入/输出。

实际运行命令与自有图片登记格式见[检测实验入口](../experiments/detection/README.md)。实现一致性只说明代码链路一致，不增加独立识别样本，也不消除公共图和现场的差异。

更新后的研究包包含20个文件。解到独立目录后确认模块来自解包目录，并实际运行6张原图，覆盖正确接受、已知错认、已知拒识、陌生拒识、多猫和漏检；逐图状态与分数和研究结果一致，错认没有被隐藏。又从126张原图重建图库，向量/阈值仍完全一致，两次检查均未导入Torch。全仓自动测试85项通过；这些是实现验证，不是新增准确率证据。
