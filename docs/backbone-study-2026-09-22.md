# 本机主干与输入尺寸对照（2026-09-22）

## 实验设置

在服务器暂不可用时，用本机 CPU 比较 MobileNetV3 Small/Large 与 160/224 输入尺寸的四种组合。所有模型均为官方 ImageNet V1 初始化，不做猫身份微调；使用分类器之前的池化特征和 L2 归一化，不使用分类 logits。Small 输出 576 维，Large 输出 960 维。

权重来自 [Torchvision Small](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html) 与 [Large](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_large.html) 的显式 `IMAGENET1K_V1`，没有使用可能变化的 `DEFAULT`。本项目仍统一采用灰色补方、PIL bilinear 和 ImageNet 归一化；即使输入为 224，也不是官方的 resize 后中心裁剪。改变架构也改变预训练权重，不能把 Small/Large 差异单独归因于参数量。

沿用原始 `edge160/protocol.json`，只对 49 个开发身份的 275 张图片提取特征，使用上一轮相同的两种 40-episode 设计。匹配固定为平均原型/余弦，不再搜索其他匹配规则。每个模型、每个图库独立用 `calibration_unknown` 校准 1% 经验阈值。本轮没有提取 test 分区特征或按 test 结果选模型。

重复图片、缺少真实拍摄场次、互联网预训练图片潜在重合等边界不变。下面都是 episode 等权宏平均；两套设计部分图片/图库相同，不能算成独立复现实验或累加样本量。严格登记配对的含义见[上一轮报告](gallery-study-2026-09-22.md)。

## 真实身份结果

| 模型 | 图库敏感性 DIR / FPIR | 严格登记配对 DIR / FPIR | ONNX 大小 |
| --- | ---: | ---: | ---: |
| Small-160，原基线 | 82.75% / 2.96% | 80.09% / 3.06% | 3.55 MiB |
| Small-224 | 82.24% / 3.02% | 84.09% / 3.27% | 3.55 MiB |
| Large-160 | 87.42% / 3.50% | 89.56% / 3.36% | 11.33 MiB |
| Large-224 | 90.18% / 2.57% | 92.10% / 2.17% | 11.33 MiB |

Large-224 是这一组中值得继续验证的精度候选，但尚未成为生产模型：

- 图库敏感性设计中，相对 Small-160 的 DIR 在 25/40 组提高、6/40 组下降、9/40 组不变；FPIR 在 18 组降低、15 组升高、7 组不变。5 猫/3 张设置的平均 FPIR 反而从 1.36% 升到 2.05%。
- 严格登记配对中，DIR 在 27/40 组提高、3/40 组下降、10/40 组不变；FPIR 在 14 组降低、18 组升高、8 组不变。平均 FPIR 的下降受少数原本误认较多的图库改善影响，不能说每组拒识都更稳。
- 敏感性设计的 DIR 第 10 百分位从 64.82% 提高到 74.87%，最高单组 FPIR 从 16.67% 降到 8.70%；这些不是置信区间，更不是现场保证。
- 仅把 Small 从 160 放大到 224，没有在两种设计中同时改善 DIR 和 FPIR。更高分辨率不是当前小模型的直接解决方案。

## 原固定开发图库的错误仍存在

Large-224 另外在原始 5 猫/每猫 3 张开发图库上生成了可重放的实际图库。111 张陌生图校准得到阈值 `0.86698928569157`：

- 已知 Top-1 和 DIR 都为 **27/29**；另外 **2/29 被接受成错误身份**，没有已知拒绝。
- 陌生误认为 **3/120**，不是校准目标 1%。
- 两张已知错认与原 Small-160 完全相同：`individuals/0004/0004_001.png` 与 `0004_034.png` 均被认成 `individuals:0294`。
- Small 的 4 次陌生误认与 Large 的 3 次中，1 次共同；换到 Large 消除原来的 3 次，同时新增 2 次。不能把总数下降解释为所有难例都解决。

因此重复图库中的较高 Top-1 不能遮盖原固定图库的真实错认，也不能将该公开图片实验当作门口验收。

## 同机 CPU 成本

Intel Core Ultra 7 155H、Windows 11、Python 3.12.10、ORT 1.20.1 CPUExecutionProvider，统一 2 线程。四种组合各启动独立进程测 3 轮，轮换测量顺序；每轮同样的 20 张原分辨率 PNG、预热 20 次、循环 100 次。

| 模型 | 特征 P50 中位数 | 三轮特征 P95 范围 | 三轮进程峰值内存 |
| --- | ---: | ---: | ---: |
| Small-160 | 1.60 ms | 2.09–2.36 ms | 114.45–116.54 MiB |
| Small-224 | 2.28 ms | 3.05–3.25 ms | 114.06–114.20 MiB |
| Large-160 | 3.28 ms | 4.47–4.81 ms | 127.03–129.36 MiB |
| Large-224 | 4.85 ms | 6.17–7.19 ms | 127.92–130.38 MiB |

本机 Large-224 的特征计算约为 Small-160 的 3 倍。PNG 读取/解码/预处理加特征的 P95 分别约 69–75 ms 与 75–92 ms；这轮为了独立比较主干没有加载图库，匹配耗时字段只是计时空段，不是匹配性能。所有时间均不含检测或相机，不能按倍数推算飞腾派。大模型是否满足飞腾派预算仍需实测。

Small-224、Large-160、Large-224 都用 5 张真实训练图通过 PyTorch/ONNX 对照，最大特征绝对误差分别为 `4.62e-7`、`3.87e-7`、`2.42e-7`。运行时、登记与缓存已按模型 metadata 读取维度，旧 Small 的训练 checkpoint 仍可严格加载。

## 复跑

```bash
python -m experiments.identity.run export --size 224 --protocol experiments/runs/edge160/protocol.json --out experiments/runs/backbone-study/small224 --threads 2
python -m experiments.identity.run export --architecture mobilenet_v3_large --size 160 --protocol experiments/runs/edge160/protocol.json --out experiments/runs/backbone-study/large160 --threads 2
python -m experiments.identity.run export --architecture mobilenet_v3_large --size 224 --protocol experiments/runs/edge160/protocol.json --out experiments/runs/backbone-study/large224 --threads 2
python -m experiments.identity.study --protocol experiments/runs/edge160/protocol.json --models small160=experiments/runs/edge160/baseline/model.onnx small224=experiments/runs/backbone-study/small224/model.onnx large160=experiments/runs/backbone-study/large160/model.onnx large224=experiments/runs/backbone-study/large224/model.onnx --methods prototype:cosine --out experiments/runs/backbone-study/development --threads 2
python -m experiments.identity.study --protocol experiments/runs/edge160/protocol.json --models small160=experiments/runs/edge160/baseline/model.onnx small224=experiments/runs/backbone-study/small224/model.onnx large160=experiments/runs/backbone-study/large160/model.onnx large224=experiments/runs/backbone-study/large224/model.onnx --methods prototype:cosine --paired-enrollment --cache-dir experiments/runs/backbone-study/development/cache --out experiments/runs/backbone-study/paired-development --threads 2
python -m experiments.identity.run evaluate --model experiments/runs/backbone-study/large224/model.onnx --protocol experiments/runs/edge160/protocol.json --out experiments/runs/backbone-study/large224/development --threads 2
python -m experiments.identity.deploy benchmark --model experiments/runs/backbone-study/large224/model.onnx --manifest data/clean.identity.jsonl --threads 2 --warmup 20 --runs 100 --out experiments/runs/backbone-study/large224/benchmark.json
```

逐 episode 预测与汇总在 `backbone-study/{development,paired-development}/`；同机三轮计时在 `backbone-study/benchmarks/`。模型和图库继续留在 `experiments/`，没有移入稳定主包。

## 本机蒸馏：已跑满 30 轮，没有取得净收益

随后冻结 Large-224，以原有 Small-160 官方预训练初始化做一次 30 轮蒸馏。本实验借鉴[相似关系保持蒸馏](https://openaccess.thecvf.com/content_ICCV_2019/html/Tung_Similarity-Preserving_Knowledge_Distillation_ICCV_2019_paper.html)的思路，但不是该论文原公式复现：本实现最小化教师/学生归一化 embedding 的批内余弦矩阵非对角项 MSE，不加入分类损失或额外投影头。

- 只用 148 个训练身份的 678 张训练图；同身份的 148 张留出图只用来选 checkpoint，没有拿开发图片做梯度训练。
- 先缓存教师在 224 输入下的原图/翻转特征；每次训练按同一次随机翻转选择相应教师目标，学生仍为 160 输入。验证不翻转，分批提特征后合并全部 148 张计算关系误差。
- AdamW、lr `1e-4`、weight decay `1e-4`、cosine、batch 32、seed `20260922`，本机 CPU 4 线程，完成 30 轮。按训练身份验证误差选中第 13 轮，没有按开发 DIR 挑轮数。
- 学生仍输出 576 维，ONNX 3.55 MiB，真实训练图导出最大误差 `5.36e-7`。模型 metadata 保留教师来源和模型校验值。

| 模型 | 图库敏感性 DIR / FPIR | 严格登记配对 DIR / FPIR |
| --- | ---: | ---: |
| Small-160 原基线 | 82.75% / 2.96% | 80.09% / 3.06% |
| 蒸馏 Small-160 | 78.27% / 2.46% | 79.08% / 2.68% |
| Large-224 教师 | 90.18% / 2.57% | 92.10% / 2.17% |

学生的陌生误认略降，但正确接受也下降，尚未把教师的收益转移到小模型。原固定开发图库中，学生 Top-1 为 28/29、DIR 21/29，错认 0/29、拒绝 8/29，陌生误认 0/120；这个单组零误认不能掩盖重复图库的不足。**本次蒸馏不升级为端侧推荐模型，也不继续按开发结果搜索轮数或超参数。** 这只能否定当前配方，不能据此断言蒸馏普遍无效。

教师的选择已经参考开发集，因此后续学生的开发结果同样是探索性比较，不是独立验证。新的现场事件仍是验收所需数据。

```bash
python -m experiments.identity.distill --teacher experiments/runs/backbone-study/large224/model.onnx --manifest data/clean.identity.jsonl --protocol experiments/runs/edge160/protocol.json --out experiments/runs/backbone-study/distilled160 --epochs 30 --threads 4 --device cpu
python -m experiments.identity.run export --checkpoint experiments/runs/backbone-study/distilled160/best.pt --size 160 --protocol experiments/runs/edge160/protocol.json --out experiments/runs/backbone-study/distilled160 --threads 2
python -m experiments.identity.study --protocol experiments/runs/edge160/protocol.json --models small160=experiments/runs/edge160/baseline/model.onnx distilled160=experiments/runs/backbone-study/distilled160/model.onnx large224=experiments/runs/backbone-study/large224/model.onnx --methods prototype:cosine --cache-dir experiments/runs/backbone-study/development/cache --out experiments/runs/backbone-study/distillation-development --threads 2
```

严格配对复跑在最后一条命令增加 `--paired-enrollment`，输出改为 `distillation-paired`。训练输出目录须为空，重训请使用新目录；已完成的训练不应覆盖。真实训练日志、30 轮历史和 checkpoint 保存在 `backbone-study/`。
