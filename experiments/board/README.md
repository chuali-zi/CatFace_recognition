# 飞腾派一键离线测试

用现有 SSDLite320 检测器和 Small160 / Large224 FP32 身份模型，在真实完整原图上采集板端数据。默认两模型分别测 1、2、4 线程，每组独立进程，模型各加载一次。固定研究图库及阈值，运行时不训练、不重新登记或校准。

## 板端运行

把准备好的 `feiteng-offline.tar.gz` 复制到飞腾派，解压后执行：

```bash
tar -xzf feiteng-offline.tar.gz
cd feiteng-offline
bash run.sh
```

全程不需要网络或摄像头。脚本先使用当前 Python 中已有的运行依赖；缺依赖时自动创建包内 `.venv`，仅从 `wheels/` 安装。系统需已安装 Python **3.10+、venv、pip**，以及 ONNX Runtime 所需的系统动态库；离线 wheel 必须匹配板端架构、Python 小版本和 glibc。当前默认准备 **ARM64 / CPython 3.11 / glibc 2.27+**，以包内 `bundle.json` 为准。可用 `PYTHON=python3.11 bash run.sh` 指定解释器。不会执行 apt、sudo 或联网安装。

运行过程中每 10 张图显示进度。默认每组先测第一帧、再预热 3 次、最后将固定的 **149 张查询原图（29 已登记 / 120 陌生）** 各测一次。全部六组完成后，终端打印结果目录和可直接拷回的 `.tar.gz` 路径。任何配置失败都会返回非零退出码，保留日志和已完成结果。

可选参数：

```bash
# 只测指定模型、线程；仍包含全部固定查询
bash run.sh --models small160 --threads 2

# 持续多轮，用于观察温度、频率和耗时随运行时间的变化
bash run.sh --passes 5

# 小规模通路检查（不作为完整识别率），WSL 验证使用这个入口
bash run.sh --models small160 --threads 2 --warmup 1 --limit 4
```

`--out /path/to/new-result-directory` 可指定新的结果目录。默认输出 `results/日期-时间/`，不会覆盖上次结果。内存按单配置子进程统计；两个模型不同时驻留。

## 拷回哪些数据

直接拷回终端打印的结果压缩包，里面包含：

| 文件 | 内容 |
| --- | --- |
| `summary.csv` | 六组配置的 DIR、FPIR、总耗时 P50/P95、峰值内存、错误数，可用 Excel 打开 |
| `summary.json` | 每组完整汇总、阈值、模型信息、CPU provider、首次推理及加载耗时 |
| `device.json`、`run.json` | 系统、CPU、内存、Python/依赖版本和运行参数 |
| `模型-t线程/measurements.jsonl` | 每张图每轮的检测框、身份预测、真值、各阶段耗时及失败原因 |
| `模型-t线程/telemetry.jsonl` | 每帧结束后的运行时间、RSS 内存、可读取的温度和 CPU 频率 |
| `模型-t线程/run.log` | 该配置的运行输出和错误信息 |

总耗时从读取文件开始，包含解码、EXIF/RGB 处理、检测、裁剪、身份特征和匹配；不含模型加载或写结果文件。`detect_ms` 含检测预处理及裁剪，`identity_and_match_ms` 仅统计实际进入身份分支的图片；无框、多框不混成“身份推理很快”。记录 CPU 时间占一个核心的百分比，可能超过 100%。温度/频率无权限或设备未提供时保存空字典；WSL 通常没有这些传感器。重复读取可能命中系统文件缓存。

DIR = 已登记查询中正确接受数 / 全部已登记查询数；FPIR = 陌生查询中被接受数 / 全部陌生查询数。无框、多框、运行错误保留在分母，另报各状态数量，不能把它们当成成功拒识。`--passes` 只重复计时，识别指标始终只取第一轮；`--limit` 的报告标记为 `limited_run`。

附带的是既有 development 查询、公开图发布者身份和对应冻结研究图库，不是新的独立测试或现场来访数据。图库中的 5 只猫不是用户自己的猫。这里采集的是目标机器实测值，WSL/x86 的速度不能作为飞腾派速度。

## 在有网电脑准备离线包

在保留本地模型和原图的完整仓库根目录运行，准备过程只复制已有模型及查询原图，不运行模型：

```bash
python -m experiments.board.prepare
```

产物为 `experiments/runs/feiteng-offline/` 及同名 `.tar.gz`。默认下载 ARM64 / Python 3.11 的运行时 wheels（含传递依赖），不下载 Torch。模型使用 `detection-cpu/onnx`、`edge160/baseline`、`backbone-study/large224` 及 `crop-study/*/fixed/detected.gallery.json`。不引入另一套检测后处理或重新选择阈值。

如板端版本不同，在有网电脑重新准备对应版本；例如 Python 3.10：

```bash
python -m experiments.board.prepare --python-version 3.10 --out experiments/runs/feiteng-py310
```

用 `uname -m`、`python3 --version`、`ldd --version` 查看板端环境。已安装运行依赖的设备可用 `--skip-wheels`；准备 WSL 验证包时可指定 `--platform manylinux_2_27_x86_64 --python-version 3.11`。输出目录必须为空，原始数据保持在原位置。
