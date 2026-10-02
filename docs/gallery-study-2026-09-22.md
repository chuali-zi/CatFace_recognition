# 登记图库稳定性实验（2026-09-22）

## 结论

固定单组开发图库的 ArcFace 优势没有在重复图库诊断中保持。现阶段继续把 **MobileNetV3-Small 160 FP32 预训练特征 + 平均原型 + 余弦分数**作为离线和板端性能参照，不把普通分类微调或 ArcFace 提升为默认生产模型。这里的“参照”不表示陌生猫拒识已经达标。

逐张取最大相似度没有形成稳定优势；第一、第二身份的相似度差（gap）主要通过大量拒绝已知猫降低误认，不能替代更有效的身份特征。它们只保留在 `experiments/identity/matching.py` 中用于复现对照，未加入端侧运行路径。

严格配对的登记数量实验表明，基线从每猫 3 张增加到 5 张有一定正确接受收益，但仍不能把陌生猫误认稳定控制在经验目标 1% 内。下一步重点是更有效的身份表征与困难猫/真实场次数据，而不是继续叠加匹配规则。

## 数据与比较方式

- 沿用 `experiments/runs/edge160/protocol.json` 的 148 个训练身份；本次只使用原有 **49 个开发身份、275 张猫脸**。没有重新划分训练身份，也没有读取 test 分区做本轮选型。
- 比较已有的未微调基线、普通分类 30 轮、ArcFace 30 轮 FP32 ONNX；未重新训练或根据这些结果挑 checkpoint。
- 每个模型对同一批图片提取一次真实 576 维特征；缓存绑定模型、模型 metadata、清单及图片路径/大小/修改时间。缓存不是合成特征。
- 每种方案在 2/5 只登记猫、每猫 3/5 张登记图、seed 0–9 上比较，每种设计共 40 个 episode。各模型和匹配方案使用相同 episode。
- 每个 episode 内，登记身份、校准陌生身份、测试陌生身份互斥；登记图与已知查询图分离。阈值只由该图库的 `calibration_unknown` 按 1% 经验目标决定，不使用该 episode 的 `unknown` 分数调阈值。
- 所有统计均先算每个 episode 的指标，再对 episode 等权平均（macro）。图片和身份跨 episode 重复，**不能把预测行累加成新增猫、独立测试样本或来访事件**。没有真实场次，公开 ImageNet 预训练与原始图片的潜在重合也未审计。

两种设计各有用途：

1. **图库敏感性设计**：每个登记数量分别从符合条件的身份中抽图库。改变登记数量可能改变合格身份集合、图库猫和陌生猫，故 3/5 张之间不是纯粹增加照片的因果比较。图库 2/5 只之间也会改变陌生猫组成。
2. **严格登记配对设计**：先要求每身份至少 6 张图，抽定每猫 5 张登记图；3 张方案使用同一组的前 3 张，另 2 张不用。两边已知查询、校准陌生猫和测试陌生猫完全相同。只在同一图库大小、同一种子内部比较增加登记图的影响；仍不把 2/5 猫之间称为纯变量比较。

## 图库敏感性结果

DIR 是“身份正确且通过阈值”，FPIR 是“陌生猫被接受为任一登记猫”；下表均为 40 个重复 episode 的宏平均百分比，不是现场总体错误率。

| 模型 | 匹配方式 | DIR ↑ | FPIR ↓ |
| --- | --- | ---: | ---: |
| 未微调基线 | 平均原型 / 余弦 | 82.8% | 2.96% |
| 未微调基线 | 逐张最大 / 余弦 | 78.1% | 2.66% |
| 未微调基线 | 平均原型 / gap | 43.6% | 1.36% |
| 未微调基线 | 逐张最大 / gap | 45.1% | 1.76% |
| 普通分类 | 平均原型 / 余弦 | 76.6% | 3.52% |
| 普通分类 | 逐张最大 / 余弦 | 68.6% | 2.84% |
| 普通分类 | 平均原型 / gap | 51.0% | 2.71% |
| 普通分类 | 逐张最大 / gap | 49.1% | 2.97% |
| ArcFace | 平均原型 / 余弦 | 72.3% | 3.77% |
| ArcFace | 逐张最大 / 余弦 | 66.6% | 2.79% |
| ArcFace | 平均原型 / gap | 56.8% | 1.81% |
| ArcFace | 逐张最大 / gap | 53.6% | 1.71% |

基线平均原型的 Top-1 宏平均约 99.3%，但 DIR 只有 82.8%；其 DIR 的 episode 第 10 百分位约 64.8%，单个 episode 的最高 FPIR 达 16.7%。这些分位数来自小规模重复抽样，不是置信区间。高排序准确率没有消除已知/陌生分数重叠，不能据此声称基本拒识已完成。

## 严格配对：增加登记照片是否有效

下面只展示平均原型 / 余弦；每个单元为同一图库规模的 10 个种子的宏平均。3→5 张的两侧查询图及陌生猫组成完全一致，每一侧独立重新校准阈值。

| 模型 | 图库猫数 | DIR：3 → 5 张 | FPIR：3 → 5 张 |
| --- | ---: | ---: | ---: |
| 未微调基线 | 2 | 80.4% → 84.3% | 3.37% → 3.21% |
| 未微调基线 | 5 | 75.3% → 80.5% | 2.74% → 2.90% |
| 普通分类 | 2 | 66.6% → 79.9% | 2.44% → 2.50% |
| 普通分类 | 5 | 54.9% → 69.5% | 3.60% → 3.95% |
| ArcFace | 2 | 78.6% → 76.1% | 3.12% → 3.56% |
| ArcFace | 5 | 53.1% → 69.9% | 4.36% → 5.00% |

配对设计每个 episode 的已知查询只有 4–26 张，校准陌生查询 94–136 张，测试陌生查询 102–140 张，仍是小样本诊断。基线 2 猫图库的 10 对结果中，DIR 有 3 对提高、7 对不变；5 猫图库有 6 对提高、2 对不变、2 对下降。

补登记图对基线的平均表现有帮助，但不是每个模型/图库都单调改善，也没有稳定降低陌生误认。该设计筛选了至少 6 张图片的身份，不能将结果推广到照片稀缺的全部猫。实际登记可先准备 5 张不同视角、清晰且来自不同时间的猫脸；本实验未证明连续拍 5 帧就有相同收益。

## 复跑与输出

在已经生成上述三个模型的完整仓库运行：

```bash
python -m experiments.identity.study --protocol experiments/runs/edge160/protocol.json --models baseline=experiments/runs/edge160/baseline/model.onnx ce=experiments/runs/edge160/trained/model.onnx arcface=experiments/runs/edge160/arcface/model.onnx --out experiments/runs/gallery-study/development --threads 2
python -m experiments.identity.study --protocol experiments/runs/edge160/protocol.json --models baseline=experiments/runs/edge160/baseline/model.onnx ce=experiments/runs/edge160/trained/model.onnx arcface=experiments/runs/edge160/arcface/model.onnx --paired-enrollment --cache-dir experiments/runs/gallery-study/development/cache --out experiments/runs/gallery-study/paired-development --threads 2
```

每个输出目录包含：

- `suite.json`：先于评分写出的完整抽样设计、各组实际图片和配对中未使用的登记图 ID。
- `summary.json`：全部候选汇总；各候选子目录的同名文件另含各图库设置的分项统计。
- `<候选>/episodes.jsonl`：每行是一整个 episode 的指标和逐查询预测。它不是原先的扁平 `predictions.jsonl`，不能整份交给 `catid evaluate`；只对其中某一行的 `predictions` 调用统一评测。
- `cache/`：真实特征缓存，可供第二种设计复用。手动改变预处理实现后应使用新的缓存目录。

所有新研究仍留在 `experiments/`；部署命令仍只接受平均原型/余弦图库，不会把 gap 阈值误当余弦阈值使用。576 维匹配分数与旧评测/端侧路径逐浮点一致并经过 JSON 往返测试，避免阈值边界因矩阵批运算改变。

本轮没有更新 test 成绩，没有板端或 3060 实测。INT8 单独排查见 [量化审计](int8-audit-2026-09-22.md)；相机和飞腾派到位后的验证边界仍见 [现场协议](camera-validation.md)与[端侧准备报告](edge-readiness-2026-09-22.md)。
