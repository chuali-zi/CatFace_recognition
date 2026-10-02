# 猫数据内容清理

在仓库根目录运行。只生成派生文件，原图和发布者身份标签不改写；身份候选必须查看猫脸裁剪与原图缩略图后才能放行。

```powershell
python -m pip install -r experiments/data_cleaning/requirements.txt
python experiments/data_cleaning/clean.py scan
python experiments/data_cleaning/clean.py propose
python experiments/data_cleaning/landmarks.py
# 查看 data/review/cleaning/faces-*.jpg，逐张记录实际决定
# 示例命令格式；索引范围必须对应刚看过的候选页
python experiments/data_cleaning/record_review.py 0 24 --reject "1:soft_face,8:partial_face_crop"
# 在完整记录后，比较跨身份最相似图片，核验或补充组级隔离
python experiments/data_cleaning/cross_identity.py
python experiments/data_cleaning/export.py
python experiments/data_cleaning/verify.py
```

`scan` 对全部 `prepared.identity.jsonl` 使用 OpenCV `haarcascade_frontalcatface_extended.xml` 生成猫脸候选框。它只做候选召回，会漏检或误检，不能用检测成功代替猫脸及身份审核。候选质量在 128×128 灰度裁剪上测量；原图最短脸边至少 96 像素，Laplacian 方差至少 100、灰度标准差至少 25。这些是本批筛选阈值，不是可迁移的质量标准。多个候选脸、已知多猫和精确冲突关联身份进入复核。按身份要求原图 / 镜像 dHash 距离、裁剪 pHash 距离都大于 8，其余只是抽稀，没有认定为精确重复。

`visual-decisions.csv` 按原图路径与实际裁剪框保存目视决定，不能用脚本批量把未看过的图设成 accept。若发现组内混猫或疑似跨身份同猫，把涉及身份及原因写入 `data/manual-group-review.csv`；export 会撤回该身份此前暂时通过的全部图，不猜测合并标签。新 proposals 的索引可能变化，记录前须查看本次生成页面。

`export.py` 只放行有对应裁剪目视通过记录且没有组级排除的图片。每身份至少两张，最多 12 张，超量时按清晰度与脸部 pHash 差异选代表。它生成 `data/clean/faces/` 无损 PNG 和四类 clean 清单，并清除这个专用输出目录中不再被清单引用的 PNG。原始 raw 不动。`verify.py` 逐张验证图片、原图对应、审核状态、排除规则、身份样本数及输入去向。

本次 2026-09-17 输出是 **1,369 张 / 247 个身份**。已记录的 2,330 张 proposals 在后续应用组级规则重跑 prepare 时保留作为本次审核快照；复建这批输出直接运行 export 和 verify 即可，不必重新 propose。全部数字见 [内容清理报告](../../docs/data-cleaning-2026-09-17.md)。

`landmarks.py` 检查全部保留关键点记录的重复点位、脸框、眼间距和脸部质量，输出 `clean.landmarks-candidates.jsonl`，仍明确标记语义审核未完成。它不会把无身份标签的关键点数据并入身份数据。

所有阈值都应结合最终输出逐张复核。记录在 `data/review/cleaning/`，模型辅助检查不代表身份的事实真值，不生成虚构的拍摄场次或可用于现场验收的划分。

## 辅助模型来源

- OpenCV 猫脸级联：[官方权重](https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_frontalcatface_extended.xml)，随本机 OpenCV 包使用，作为快速候选定位器。
- 可选 ECPose：[MeowID-Base 官方发布](https://huggingface.co/RicePasteM/MeowID-Base)，发布页标注 Apache-2.0。下载 `artifacts/MeowID-Base/ecpose.onnx` 至本目录 `models/`；发布的部署参数保存为 `models/deployment.json`。ONNX SHA256 为 `aed358530fd8d0769c1dad678272c169dc3ba12c3c54b2979ead24eefd758d67`。输入 RGB、640×640、ImageNet 均值/标准差；实际模型 `orig_target_sizes` 输入为 float32，顺序宽、高。用 `clean.py scan --detector ecpose` 可运行辅助关键点定位，当前本机 CPU 的试跑速度不适合用它承担全量主流程。没有将其选为最终嵌入式识别模型。

本轮没有训练或蒸馏。
