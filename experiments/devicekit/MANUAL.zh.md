# 科研测试手册

本包不控制门，不依赖云业务/LLM。检测与身份核心代码/模型保持冻结。v3提供独立ARM运行时；v4补齐冻结图片/图库与`bash run-all.sh`统一入口，自动执行两模型×三线程×三轮并打包一个结果ZIP。源码来源与哈希见SOURCE_MANIFEST.json。

| 模式 | 真正前提 | 结果含义 |
|---|---|---|
| preflight | 标准库、源码封存清单 | 环境/包完整性，未加载模型 |
| model-check | 检测+选中身份ONNX/元数据、NumPy/Pillow/ORT | 实际CPU加载/图接口；未识别图像 |
| enroll | 上述+真实身份/场次已确认的登记/陌生校准图 | 本地建库与校准；非现场准确率 |
| recognize | 上述+匹配且有阈值证据的图库和实际单图 | 一帧诊断；不产生准确率/P50/P95 |
| field | 上述+独立场次/陌生身份查询及真值 | 按真实分母诊断，非生产认证 |
| historical/smoke | 冻结图库/协议/证据及149原图 | 公开图历史诊断；v4包含，v3不含 |
| camera | 现场同意、cv2、真实设备/现有权限 | 只抓一帧后丢弃，不自动打开 |

## 模型契约

检测RGB float32[1,3,320,320]，内部归一化，boxes[3234,4]/scores[3234]；score0.7、topk300、NMS0.55，映射到原图。单框head裁剪，多框unusable，无框no_cat。身份EXIF/RGB、灰128居中方形填充、bilinear缩放、/255、ImageNet mean/std；Small160为576维、Large224为960维，prototype cosine。

score不是概率。模型/检测配置/裁剪/登记样本改变须重新校准；不能复制历史阈值。图库校准证据与模型、元数据SHA必须匹配。原sidecar逐字保留，没有伪造新导出或来源。

## 依赖和设备

requirements-device.txt保留v2固定11项，其中NumPy1.26.4、Pillow11.1.0、ORT1.20.1、psutil6.1.1。本机x86_64 Python3.11已核验安装/导入；推理不需要PyTorch或GPU。camera另需cv2。

2026-10-18板端已提供Ubuntu20.04.6、aarch64、glibc2.31、系统Python3.8.10。修复后的包内ARM64 CPython3.11.17离线安装、固定依赖导入和两模型CPU加载已PASS；实际图片推理与相机尚未执行。离线准备可用experiments.devicekit.dependencies download的显式--arch/--python参数，必须匹配真实目标环境。

## 用途与回传

用户已要求本人接收完整ZIP并用于同学非商业大创科研测试。本包保留完整来源/许可通知，不赋予公开或商业权利。distribution_approved=false只记录没有不受限分发授权；不能拿它作为技术推理条件，也不能改成true冒充审阅。详细适用范围与上游权重边界见LICENSE_REVIEW.zh.md。

机器报告记录真实stage状态、样本数、线程/预热、模型/包/输入SHA。回传默认三个汇总文件；private可能含标签、路径、原始日志，不能打包全目录。单帧/加载检查不写现场准确率；本机计时不外推飞腾派。

## v3运行时与来源

使用包内QUICKSTART步骤；系统Python3.8仅启动标准库诊断/bootstrap，不能运行身份推理。已有Pillow7.0不具备当前Image.Resampling接口，不沿用系统依赖。新运行时位于.runtime/python，不改系统Python、APT、驱动或用户组。安装过程校验包清单、runtime-support清单、上游SHA256及wheel锁定哈希；不联网。若失败，报告真实异常类型/原因，不声称离线兼容已验收。

运行时来自[python-build-standalone固定发布](https://github.com/astral-sh/python-build-standalone/releases/tag/20261013)，原压缩包保留61项名称含license的许可条目（包含于运行时，不另行删减）；[项目运行要求](https://gregoryszorc.com/docs/python-build-standalone/main/running.html)列出动态库边界。[ORT1.20.1官方PyPI](https://pypi.org/project/onnxruntime/1.20.1/)提供CPython3.11 ARM64 glibc2.27+ wheel，本次Pillow wheel另要求glibc2.28+。实际字节来源URL/大小/SHA256及11个wheel列表见runtime-support/RUNTIME.json和MANIFEST.json。包只面向已声明非商业研究测试，不扩大模型/数据分发权利。

## 报告与复现

详见docs/device-test-results.md。preflight会列出具体平台差距、依赖MISSING/PIN_MISMATCH与恢复命令；这些版本是安装元数据，不冒充导入成功。model-check单独记录图加载、CPU provider、实际线程，并明确inference_executed=false。真实身份/时延必须由真实图片分支单独验证，不能把preflight/model-check/bootstrap PASS当成识别验收。
