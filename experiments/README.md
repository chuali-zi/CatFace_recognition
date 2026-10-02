# 识别实验

当前已实现 [identity/](identity/README.md) 中的 MobileNetV3-Small 160 猫脸身份候选：预训练池化特征、简单分类微调、ONNX 导出、静态量化对照、开集评测、CPU 性能测量与打包。尚未选为生产模型。研究问题和顺序统一在 [路线整理](../docs/research.md)，评价方式在 [评测协议](../docs/evaluation.md)。

现有 [data_cleaning/](data_cleaning/README.md) 用于猫脸内容清理、逐张决定和派生数据导出；`data_audit/` 保存前置审计脚本与发现。这些工具不构成身份识别模型，清理结论见[报告](../docs/data-cleaning-2026-09-17.md)。

有实际实验时创建一个有含义的候选目录，放可运行脚本和简短 README，说明数据、输入、模型来源、依赖、命令和当前结论。共享的数据与指标直接调用 `catid.data` 和 `catid.evaluation`。

权重、缓存、逐查询预测和指标放 `experiments/runs/<run-name>/`；该目录不进入版本控制。基础包不预装 PyTorch、ONNX Runtime 或某个候选的依赖，由候选说明实际需要什么。

不提前建立空训练框架或插件系统。路线确定后将必要实现整理进 `src/catid/`，删除被放弃的实验脚本和产物，并更新路线说明。
