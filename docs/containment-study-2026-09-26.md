# 重复候选框的固定包含抑制对照

## 结论

已实现一个可选的研究配置：在原有 score 0.7 / NMS 0.55 / 最多 10 框之后，对高度包含的候选仅保留高分框。固定 `intersection / min(area) >= 0.9`，不融合框、不改获胜框坐标，也没有搜索阈值。

训练与开发数据中共 4 张同头重复框恢复为单框，已标注目标覆盖不变。已目视检查的 7 张真实多猫图仍保留多框。此收益有限，尚不能证明强遮挡、多猫头嵌套时一定安全；**不改默认检测配置、不替换原研究包**，另提供显式开启的候选包。

## 使用的数据

- CatFLW train：1,447 图，本轮重新运行冻结 ONNX 检测器。
- CatFLW development：312 图，使用已冻结的 `ssdlite-selected-development.jsonl`，实际配置为 score 0.7、NMS 0.55。不要误用最初 score 0.3 的 `ssdlite-development.jsonl`。
- 旧 identity train/validation：678 + 148 = 826 张原图，包含检测失败。不能用已经排除失败的 head-protocol 811 张替代。
- 旧 identity development：275 张原图。

没有读取 CatFLW test、identity test 或上一轮 fresh-identities 图片来选择、调整此规则。原模型权重和输入处理不变。CatFLW train 是检测训练数据，只能检查行为；完整作者标注通常不穷尽所有猫头，不以其额外预测数推导误报率。

## 实际结果

| 数据 | 单框输入：原始 → 候选 | 多框输入：原始 → 候选 | 无框输入 | 已标注目标覆盖：原始 → 候选 |
| --- | ---: | ---: | ---: | ---: |
| CatFLW train 1,447 图 | 1,421 → 1,423 | 6 → 4 | 20，不变 | 1,421 → 1,421 |
| CatFLW development 312 图 | 298 → 300 | 5 → 3 | 9，不变 | 299 → 299 |
| 旧身份 train/validation 826 图 | 814 → 814 | 2 → 2 | 10，不变 | 无完整脸框真值，不计 |
| 旧身份 development 275 图 | 273 → 273 | 1 → 1 | 1，不变 | 无完整脸框真值，不计 |

人工 exhaustive development 64 图，严格 IoU≥0.5 一对一匹配：两种配置均为 TP=62、FP=1、FN=2。该子集在 score 0.7 下本来没有多框，所以数字不变仅是回归检查，不代表覆盖了困难多猫输入。

所有 14 张原多框输入另行查看：6 张同头重复框、7 张真实多猫、1 张猫脸倒影。6 张重复框中消掉 4 张，还有 2 张未达到固定 0.9 包含条件；未为了消掉它们降低阈值。倒影位置与实体脸分离，仍按多候选不可用处理。

恢复的 4 图：

- train `00000400_021.png`：草地黑白猫，去掉包含头和部分身体的低分大框。
- train `00000420_018.png`：植物中的歪头虎斑，去掉重复局部小框。
- development `00000060_001.png`：灰色幼猫重复框。
- development `CAT_01_00000130_021.png`：黑白幼猫嵌套框。

后两例保留下来的高分框仍含部分身体。去重只是解除候选数量歧义，不等于脸框定位更准确。真实多猫只覆盖少量常见姿态，尚未验证严重重叠/嵌套猫头；本轮也没有真实无猫负例或门口事件数据。

## 已验证的运行路径

`onnx._select` 只在 metadata 显式提供 `containment_threshold` 时开启；缺少该字段或值为 null 时保持原行为。候选使用与原检测器完全相同的 ONNX 权重，仅旁侧配置不同。

- 将可选配置接到真正 Detector 后，重新推理 312 张 development 原图，框和分数逐项等于离线对照结果：312/312。
- 用原来的 126 张登记/校准原图重新生成候选图库：15 张登记成功、111 张计划陌生校准中 109 张成功；向量及阈值与原配置完全相同。这是因为这批图片没有触发包含抑制，不能推广为其他图库一定不变。
- 新旧配置的 metadata 绑定不同，直接拿旧图库配候选检测器会报错，必须重新登记/校准。
- 完整重放中，4 张恢复图均从 `unusable` 进入身份分支并输出 `unknown`；这是流程验证，CatFLW 不提供这些图与登记身份的个体真值，不能把它计为陌生识别准确率。
- 旧身份数据的 3 张真实多猫图仍返回 `unusable`；`0004_048.JPG` 仍为 `known`，匹配分数保持 0.9030434349131486。上述重放进程没有导入 Torch。

现有旧身份开发图的框/分数均不变，没有据此制造新的身份识别收益。Small/Large、陌生拒识与飞腾派实测的前述结论不变。

## 复跑与候选使用

```powershell
python -m experiments.detection.suppression_study --out experiments/runs/containment-study --threads 2
python -m experiments.detection.containment --model experiments/runs/detection-cpu/onnx/model.onnx --out experiments/runs/containment-study/detector
python -m experiments.detection.enroll --manifest experiments/runs/crop-study/enrollment-manifest.jsonl --detector experiments/runs/containment-study/detector/model.onnx --model experiments/runs/backbone-study/large224/model.onnx --crop-mode head --threads 2 --out experiments/runs/containment-study/gallery.json
```

研究目录及图库拒绝覆盖已有输出，复跑要用新的目录。研究包生成命令：

```powershell
python -m experiments.identity.package --mode edge --model-dir experiments/runs/backbone-study/large224 --gallery experiments/runs/containment-study/gallery.json --detector-dir experiments/runs/containment-study/detector --out experiments/runs/catid-edge-containment-research.tar.gz
```

候选解包后的登记、重放命令与原检测研究包相同；其 detector metadata 已显式开启包含抑制。它仍只是公开图研究模型，附带图库不是用户自己的猫，数据授权与现场/飞腾派边界没有改变。

逐图原始框、候选框和分数在 `experiments/runs/containment-study/*.jsonl`；14 张对照图、逐图视觉结论分别在 `review/` 和 `visual-review.json`。`summary.json` 保存完整分母，`runtime-development.jsonl` 保存真正运行时复核输出。
