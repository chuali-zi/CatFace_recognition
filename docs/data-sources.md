# 数据来源核对（2026-09-16）

本次目标是准备研究数据，不进行训练。已有目录标签保留来源前缀；它们不是重新人工确认的猫身份。下面的许可来自发布页或官方 API 元数据，实际应用还需遵守原发布者条款。

| 来源 | 用途与本次处置 | 发布许可 |
| --- | --- | --- |
| [Cat Individual Images](https://www.kaggle.com/datasets/timost1234/cat-individuals) | 已有，主身份研究集；本地原清单 13,106 张、510 个标签 | CC BY 4.0 |
| [HelloStreetCat Individuals](https://www.kaggle.com/datasets/tobiastrein/heellostreetcat-individuals) | 已有，固定投喂场景；视角与相邻帧相关，不能拿随机图片划分证明现场效果 | CC BY-NC 4.0 |
| [Cat Re-Identification Image Dataset](https://www.kaggle.com/datasets/cronenberg64/cat-re-identification-image-dataset) | 已有，暂不纳入可用身份子集；原图抽查有大量保险广告，原始 info.json 中 30 个 name 为“最新の譲渡会情報”，部分标签来源是列表页，单纯去重不足以修好身份 | Kaggle 标注 MIT；不能据此假定抓取图片的所有权已解决 |
| [CAT](https://www.kaggle.com/datasets/crawford/cat-dataset) | 已有，9 点关键点任务；镜像目录去重，越界标注保留坐标并隔离 | 本轮未确认原图授权；不凭镜像许可推断 |
| [CatFLW 官方仓库](https://github.com/martvelge/CatFLW) / [下载](https://www.kaggle.com/datasets/georgemartvel/catflw) | 本次补充，发布者说明 2,079 张、48 点和脸框；无身份标签。包含 CAT 来源图，必须与已有 CAT 联合查重 | CC BY-NC 4.0 |
| [PetFace 官方仓库](https://github.com/mapooon/PetFace) | 适合后续身份数据和教师候选。官方要求机构邮箱、英文申请并人工批准，不能直接无登录下载数据。本次未提交申请 | 数据、代码、权重仅非商业研究 |
| [Petfinder Cats](https://zenodo.org/records/6656292) / [作者处理代码](https://github.com/MarQuisCheshire/Pets-Face-Recognition) | 开放下载，但 6 个压缩包共约 36.62 GB；领养档案不一定是单猫身份，仍需核对。此次先处理现有集，不重复引入大量未经审核的网络图 | Zenodo API 标注 CC BY 4.0 |
| [PPGNet-Cat](https://github.com/victorcaquilpan/PPGNet-Cat) | 2025 年野猫重识别研究；主要训练数据私有，仓库演示图不能当作新大数据集 | 按仓库及来源分别核对 |

没有把品种分类、猫狗分类或合成猫图当作真实个体身份新增数据。WildlifeDatasets 中的 CatIndividualImages 也是已有来源，不重复下载来虚增规模。

## 教师模型留待下一阶段

[PetFace](https://github.com/mapooon/PetFace) 提供猫 ArcFace 等官方权重入口，可以后续在 remote RTX 3060 上核对依赖与显存后比较。[MeowID-Base](https://huggingface.co/RicePasteM/MeowID-Base) 模型卡公开了脸部/全身双分支及 PyTorch、ONNX 产物；其 TensorRT 文件针对作者 RTX 3090 构建，不能假定可直接给 3060 使用。两者都没有在本项目运行或验证准确率。

本次不下载训练环境和教师权重，不启动 SSH/GPU 任务。模型选择与蒸馏等数据审核完成后再开展。
