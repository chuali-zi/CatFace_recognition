# 统一检测裁剪的身份训练对照（2026-09-26）

本轮只检验一个问题：在网络、初始化、训练预算和样本身份完全相同的情况下，用部署时的检测猫头裁剪训练，能否改善未见猫身份识别？不修改检测器，不修改开发图库采样，不使用identity test选择模型。

## 为什么做这组对照

上一轮Large224/head共同39组实验中，陌生接受出现121次，但只涉及42张不同图片、14个身份；face_core为132次、61图、21身份。已知拒识分别为85次/21图/8身份和60次/22图/8身份。重复出现不是新的独立样本，不能直接按这些次数增大某些开发图片的训练权重；本轮开发图片完全不参与训练。

目视复核4张代表性错误crop：`0004_001`的极近距离紧裁剪与左右受光差，`0412_011`的耳朵/头顶缺失，`0446_041`的项圈、室内背景与眼部明暗差，`0429_003`的轻微侧转和蓝色织物背景。这些并非明显无效图片，不删除或改标签；可见因素只能解释潜在风险，不能从图像外观断言错误原因。

## 匹配训练样本

原协议train 678图、validation 148图，共826图/148身份。冻结SSDLite score=0.7，仅对这两部分原图离线检测一次：814图单框，10图无框，2图多框。对单框训练图额外核对既有审核face框中心落在预测框内，以确认训练标签指向的目标；没有因此再排除图片。此核对只用于训练标签有效性，不输入实际推理，不修框，也不用于删除开发查询。

仅保留同时有成功train图和原validation图的身份。两臂最终都使用145身份、train 666图、validation 145图，共811图。排除的3个身份是0250、0256、0357；除了12个无框/多框输入，另有3张成功图因所属身份无法组成原训练/验证对而一起排除。原图未删除，未补图或重新抽validation。

目视抽查排除原因：`0357_010`和`0060_012`确实都有两只猫；`0250_009`有侧向/家具遮挡的灰白虎斑猫；`0256_003`则是清楚的近距离单猫，属于明显漏检。0256共6张原train/validation图全部无框，排除它会使训练池偏向检测器容易命中的身份，不应解释成原图或身份本身无效。两臂匹配可以控制样本差异，但不能消除这种选择偏差。

- legacy臂：使用上述相同sample ID的原审核PNG。
- head臂：读取原图，在EXIF转正后按离线检测整数xyxy框裁剪。
- 两臂都使用Small160、ImageNet V1初始化、分类交叉熵、30轮、batch32、AdamW lr=1e-4/weight_decay=1e-4、cosine学习率、seed20260922、CPU4线程。仅随机水平翻转，缓存确定性预处理tensor。
- 都按各自训练身份的validation loss选择checkpoint；开发身份不参与训练或选epoch。
- development/test的身份元数据保留用于隔离检查，只有development最终运行推理；没有运行test原图。

离线协议保存`crop_box`而不另存大幅PNG，训练不会再跑检测器。训练数据读取、导出真实图片一致性检查共用EXIF/RGB/裁剪实现，推理图结构仍是原来的576维Small160特征网络，不增加端侧分支或分类头。

## 训练与导出

head臂已完成30轮，按训练身份验证损失选第29轮：validation loss 2.7334867、分类accuracy 50.34%。从数据缓存/模型初始化到30轮结束的墙钟时间312.74秒（不含Python/Torch导入）。这只是训练身份内部分类，不是未见身份识别成绩。

head ONNX为3.5458 MiB，576维，图结构不增加新推理层。5张真实训练原图按各自离线检测框裁剪后，Torch/ORT输出最大绝对差5.59e-7，最小cosine为1.0。后续图库仍须为这份新权重重建和校准。

legacy臂也完成30轮，选第30轮：validation loss 2.7166903、分类accuracy 53.10%，同口径墙钟339.64秒。其ONNX同为3.5458 MiB，5张真实审核PNG的输出最大绝对差1.02e-6、最小cosine为1.0。两臂顺序运行，没有同时训练；这些本机CPU训练时间不是飞腾派推理速度，也不是受控性能基准。验证损失是在各自裁剪输入上计算，不能据两者接近就推断未见身份能力相同。

训练数据crop_box的EXIF坐标、缓存/非缓存路径、导出输入和批量特征读取都已核对，完整自动测试88项通过。

## 完整原图开发结果

两臂都只运行原development的275张原图。与上一轮逐字段核对，275条检测记录（框、分数、裁剪坐标与状态）完全一致，仍是273单框、1多框、1漏检。所有模型使用检测head图重新登记、校准和查询；下表的legacy训练臂不是“旧登记图+新查询”的mixed模式。

40个配对episode中仍有1个原样登记失败。四套结果使用完全相同的39个有效episode，且逐episode的known、unknown、calibration_unknown样本顺序、数量一致。目标校准FPIR=1%，各模型各图库独立选阈值；失败查询保留分母，不算成功拒识。

| 身份特征 | 已知Top-1（不拒识） | 正确接受DIR | 陌生误认FPIR | 可用陌生查询条件FPIR |
| --- | ---: | ---: | ---: | ---: |
| 未微调Small160 | 97.31% | 72.72% | 1.34% | 1.35% |
| 旧crop训练Small160，同样本30轮 | 92.06% | 42.22% | 2.39% | 2.41% |
| 新head训练Small160，同样本30轮 | 97.78% | 68.82% | 2.86% | 2.87% |
| 未微调Large224，额外精度参照 | 99.23% | 82.87% | 2.41% | 2.43% |

以上是逐episode宏平均，不是把重复图片合成更多独立样本。Large224不属于同结构的训练变量对照。

统一head训练比旧crop训练恢复了26.60个百分点DIR，但相比不微调Small160仍下降3.90个百分点，FPIR增加1.52个百分点。逐episode看DIR为10组改善、22组变差、7组相同；FPIR为18组降低、11组升高、10组相同，少数组的大幅恶化使宏均值变差，不能只数改善组数。

例如seed4的5猫/3登记图设置，新head模型陌生接受35/138（25.36%），原Small160为4/138（2.90%）；对应5登记图设置为30/138（21.74%），原模型6/138（4.35%）。这两组配对复用了相同查询，不是两份独立坏样本证据；它们直接表明1%校准目标没有稳定迁移到不同陌生身份。

原固定5猫×3图图库的结果：

| 特征 | 已知正确接受 / 29 | 已知错认 / 29 | 已知拒识 / 29 | 陌生接受 / 120 |
| --- | ---: | ---: | ---: | ---: |
| 未微调Small160 | 19 | 0 | 10 | 0 |
| 旧crop训练Small160 | 14 | 4 | 11 | 6 |
| 新head训练Small160 | 21 | 1 | 7 | 0 |
| 未微调Large224 | 24 | 1 | 4 | 0 |

固定图库新head比未微调Small多接受2张，但增加1个错认，且没有在重复图库中保持收益，所以不能选这一组作为升级依据。固定图库的校准仍为计划111张、可用109张。

## 结论与实现复核

本轮不替换现有研究包和模型候选，不把两份分类微调权重提升为部署默认。保留未微调Small160速度参照、Large224精度参照；两者也仍未通过现场/陌生拒识验收。

新head特征Top-1接近98%，但DIR只有约69%，说明“在已登记猫里排对第一”与“能够接受且可靠拒绝陌生猫”并不是同一问题。这不能通过删难例、放低阈值或在已曝光test上调参解决。本轮的有限证据是：统一裁剪必要，但这套小数据分类微调没有形成稳定净收益，不能据此断言所有训练路线无效。

使用新head模型与其重建图库另做6张完整原图的实际回放，覆盖正确接受、已知错认、已知拒识、陌生拒识、多猫和漏检。状态与研究预测一致，分数差小于1e-12，未导入Torch。其中`0004_051`仍被错认为0294，`0004_048`被拒识；错误没有被单张演示隐藏。

开发缓存、三种裁剪组合的逐episode结果、固定图库、训练history及两个ONNX保存在本轮runs目录，可复查此次受控实验。它们不是新的独立测试集；没有增加现场图片，也没有测飞腾派速度。

## 复跑入口

```powershell
python -m experiments.detection.training_crops --manifest data/clean.identity.jsonl --protocol experiments/runs/edge160/protocol.json --detector experiments/runs/detection-cpu/onnx/model.onnx --out experiments/runs/head-adaptation/prepared --threads 2
python -m experiments.identity.train --manifest data/clean.identity.jsonl --protocol experiments/runs/head-adaptation/prepared/head-protocol.json --out experiments/runs/head-adaptation/head --epochs 30 --batch-size 32 --threads 4 --size 160 --device cpu --cache-images --loss cross-entropy
python -m experiments.identity.train --manifest data/clean.identity.jsonl --protocol experiments/runs/head-adaptation/prepared/legacy-protocol.json --out experiments/runs/head-adaptation/legacy --epochs 30 --batch-size 32 --threads 4 --size 160 --device cpu --cache-images --loss cross-entropy
python -m experiments.identity.run export --checkpoint experiments/runs/head-adaptation/head/best.pt --manifest data/clean.identity.jsonl --protocol experiments/runs/head-adaptation/prepared/head-protocol.json --out experiments/runs/head-adaptation/head --size 160 --threads 2
python -m experiments.identity.run export --checkpoint experiments/runs/head-adaptation/legacy/best.pt --manifest data/clean.identity.jsonl --protocol experiments/runs/head-adaptation/prepared/legacy-protocol.json --out experiments/runs/head-adaptation/legacy --size 160 --threads 2
python -m experiments.detection.crops --detector experiments/runs/detection-cpu/onnx/model.onnx --models head_ce=experiments/runs/head-adaptation/head/model.onnx legacy_ce=experiments/runs/head-adaptation/legacy/model.onnx --protocol experiments/runs/edge160/protocol.json --crop-mode head --out experiments/runs/head-adaptation/development-cache --threads 2
python -m experiments.detection.crop_study --cache-dir experiments/runs/head-adaptation/development-cache --protocol experiments/runs/edge160/protocol.json --model-name head_ce --out experiments/runs/head-adaptation/evaluation-head
python -m experiments.detection.crop_study --cache-dir experiments/runs/head-adaptation/development-cache --protocol experiments/runs/edge160/protocol.json --model-name legacy_ce --out experiments/runs/head-adaptation/evaluation-legacy
```

准备目录需为空，已有结果无需覆盖。head协议的path指向原图，crop_box是在EXIF转正后坐标；本轮必须保有相应原图，只有审核PNG的旧身份服务器包不能直接完成此训练。不要对head-protocol直接调用普通crop-only evaluate并误称完整画面结果：其development行保留原协议审核PNG，完整原图评测使用上面的detection.crops/crop_study入口。

训练和评测产物位于 `experiments/runs/head-adaptation/`，仅为公开图片级研究，不提供现场来访事件、夜间、无猫误报或飞腾派速度保证。
