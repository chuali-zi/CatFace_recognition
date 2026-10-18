# 2026-10-18 飞腾派报告与运行环境修复

最新进展：随后收到的v3报告已通过已有Python3.11环境的preflight及两模型CPU加载；v3自带bootstrap失败的问题与修复见本文末尾。下面的首轮记录保留原有验证范围。

用户确认从板子取得的原ZIP为`20261018T052044Z-bc3ea9b0.zip`，3940字节，SHA256 `ecdfbd8a8364dbfc5349ecdaa75433ab9ee5c8204269aaf9bab88393232093f8`。本机已保留原件、安全解压并核验三文件报告/事件一致性；未执行附件指令或脚本。此前Library helper的Windows xattr失败已由用户手动放入本机文件解决，不再依赖该下载路径。

## 实际结果

模式仅为preflight。系统记录Ubuntu20.04.6 LTS、内核`5.10.209-phytium-embedded-v2.2`、aarch64、4逻辑CPU、内存3869.5MiB、glibc2.31、CPython3.8.10。用户确认设备来源；报告没有具体硬件型号字段，operator_execution_context为unknown，不能从架构单独推断所有硬件已验证。

| 阶段 | 实际状态 | 含义 |
|---|---|---|
| package_integrity | PASS | 封存文件校验通过；原始ZIP哈希字段未填写。 |
| preflight | BLOCKED / E_PLATFORM | 旧脚本要求CPython3.10–3.12；报告为3.8.10。Linux/ARM64/glibc2.31本身满足候选要求。 |
| assets、dependencies、model_load、historical、image_recognition、enrollment、camera | 全部SKIP（7项） | 未请求/未执行，不是通过，也没有模型错误或图像成绩。 |

没有FAIL阶段、没有模型加载日志或性能测量；recognition_validated/hardware_validated均false，三种model_sha256及asset_manifest_sha256为空。依赖安装元数据为NumPy1.24.4、Pillow7.0.0、ORT缺失、psutil5.5.1；preflight没有做实际导入检查。Pillow7也不具备当前预处理使用的Image.Resampling接口，不能仅放宽Python门槛就继续沿用系统依赖。

## 与旧x86验证的差异

旧交付是`delivery/catface-test-kit-20261013/CatFace-完整测试包-含真实模型.zip`，27433880字节、129条目，SHA256 `7143484ec2c1a95c87ea589e7d1aebcd70ea31003d431a8c052a0997ffe30309`；本机CRC和128文件封存哈希均通过。旧x86记录来自同机WSL/AlmaLinux9.8、Python3.11.13、glibc2.34，固定NumPy1.26.4、Pillow11.1.0、ORT1.20.1、psutil6.1.1，实际加载与研究图诊断曾执行。其单图、4查询smoke和149查询记录有不同验证范围，均不能移植为此次ARM成绩。

旧包包含SSDLite320检测器与Small160/Large224 FP32候选，没有照片、旧图库或ARM wheels。当前仓库51个已有实验/主模块文件与旧包逐字节一致；20个devicekit文件原仅存在于交付副本，本次按已封存旧包哈希增量恢复。旧副本上游commit不在本机对象库，来源链明确记录为旧包恢复后本地补丁，不伪称原历史已恢复。

## 修复范围

- 保持研究模型、阈值和检测/身份算法冻结，继续要求CPython3.10–3.12；不尝试以旧Pillow/未装ORT的Python3.8运行识别。
- 新增标准库平台逐项诊断和依赖版本清单。报告schema3明确E_PYTHON_VERSION、actual/required/remediation；缺模块/动态库错误只导出白名单分类。model-check记录真实CPU provider、线程及inference_executed=false。
- 提供固定官方ARM64 CPython3.11.17及11项匹配wheels，按官方SHA256SUMS/PyPI哈希核验；锁文件`--require-hashes`用于包内离线安装。系统Python3.8可启动bootstrap，安装仅落在包内.runtime，不联网、不sudo、不替换系统解释器或libc。完整运行时保留上游许可文本。
- 修复内部shell入口切目录后的模块路径，以及Windows生成清单时的反斜杠。新增完整包生成器，按源码/许可/模型/运行时白名单构建到新目录；不包含照片、图库、秘密或旧验收ZIP。
- 原始模型SHA保持：detector `5c15910dd27108d6378257518367254af943ae3e97084cd2730367bb53205694`；small160 `0ac37dcc71f8575e3a9d406287095169a3f3100bdac806b58620afc23fe44dd6`；large224 `1518d327a2278a95143d01f011045aed3c08ee8c66121fd8a4231ed3321c0467`。

## 同学必须复测

按新包QUICKSTART在新解压目录依次执行：旧系统解释器preflight（预期明确Python版本BLOCKED）、`python3 -m experiments.devicekit.bootstrap`（预期离线安装/导入PASS）、新preflight、两模型model-check。保留每次三文件ZIP及SHA和硬件型号。bootstrap失败保留实际原因，不能宣布离线兼容；model-check成功也仍没有图片识别。

之后只有具备真实登记、陌生校准和独立查询场次时才运行enroll/recognize/field；更换模型需分别建库。没有原149图/旧图库时historical/smoke缺件BLOCKED正常。实际识别率、首帧/持续性能、内存温度、ARM指令/系统动态库和可选相机均待目标设备验证；不操作猫门、门锁或GPIO。

完整结果协议见[设备结果协议](device-test-results.md)。本机回归与新包验证记录另行附于交付核验文件，不将其表述为ARM验收。

## 本机验证边界

Windows相关回归完成23通过、1项符号链接权限跳过；增加文件系统用例后，devicekit专项复跑12通过、1跳过。WSL x86_64/CPython3.11固定依赖实际验证了根/内部shell入口、两种真实模型CPU加载、两张已有私有研究图的单帧known/unknown输出，以及历史数据缺件BLOCKED、非ARM环境bootstrap拒绝安装。报告ZIP三文件、JSON/events一致性和SHA均核验。这些都不是ARM识别率或时延验收。

独立运行时tar有4907成员；安全条目检查与许可文本保留已完成。完整解压在本机Windows共享盘遇到上游terminfo文件的大小写冲突；v3因此增加提取前文件系统检查，明确要求板子Linux区分大小写目录，不强行覆盖冲突文件或删除上游组件。内部符号链接处理用不执行任何二进制的最小tar另行验证。实际完整ARM运行时安装、导入和模型推理仍须同学在板子执行。

## v3回传与符号链接解包修复

本次附件`新建文件夹 (3).zip`含以下三份报告及各自SHA256文件。三份内层ZIP均核对SHA256、ZIP CRC及三文件结构，读取报告而未执行附件内容。

| 报告 | 实际结果 |
|---|---|
| `20261018T071408Z-bootstrap-8196b11e` | 系统Python3.8.10启动bootstrap后BLOCKED，创建`python/share/terminfo/1/1178 -> ../a/adm1178`时报FileNotFoundError。 |
| `20261018T071658Z-30a57239` | 已有Python3.11.17环境下preflight PASS，固定NumPy/Pillow/ORT/psutil版本匹配。 |
| `20261018T071735Z-74bc8127` | 依赖导入与Small160/Large224的CPU ONNX session及接口检查PASS，两个模型均明确`inference_executed=false`。 |

三份报告来自Ubuntu20.04/aarch64/glibc2.31，仍没有真实图片推理、识别率或耗时验收。后两份使用的已有环境没有证明bootstrap成功；两个身份模型及检测器哈希与冻结模型一致。

根因是上游运行时tar的4,907个条目中没有显式目录条目：3,858个普通文件、1,049个符号链接。旧提取器仅在写普通文件时创建父目录，因此像`terminfo/1`这种仅含链接的目录一直不存在；Linux允许链接目标暂时不存在，但要求链接自身的父目录存在。

修复仅在创建符号链接前补上`target.parent.mkdir(parents=True, exist_ok=True)`，保持先验证条目、提取普通文件、最后建立内部链接的顺序。未更换运行时、wheels、模型或识别算法。

本机WSL/Linux区分大小写的临时目录中，用旧版提取器复现了同一`../a/adm1178`错误；修复版完整提取4,907个条目，逐项核对所有普通文件SHA256/权限和全部符号链接目标，原报错链接可正常读取目标内容。过程未执行ARM二进制。新增回归覆盖无显式目录条目、链接出现在目标文件之前的link-only目录；Windows上无需符号链接权限也能检查创建父目录的前置条件。

重新交付使用新的解压目录，按QUICKSTART先bootstrap，再preflight和两模型model-check；不得以本机解包通过宣称ARM安装或图片识别通过。此前现场报告和已有环境保留。
