# 同学复测：Ubuntu20.04 / ARM64 / 系统Python3.8

2026-10-18首份报告在系统Python3.8下preflight受阻。最新v3报告已用原有Python3.11.17环境完成preflight及Small160/Large224的CPU模型加载；尚未执行图片识别。v3自带离线安装曾在`terminfo/1/1178`处因缺少符号链接父目录失败，本修订已补上目录创建。

请把**新版完整ZIP**解压到板子上新的可写、区分大小写的Linux目录（如home/ext4），进入含run.sh的包根目录。不要覆盖旧包；不要直接在NTFS/FAT/Windows共享盘安装运行时。包内已含真实冻结模型、ARM64 CPython3.11.17压缩包和固定依赖wheels，不含照片、旧图库或现场身份数据。

```bash
# 可先保留系统解释器的诊断报告，预期Python3.8下BLOCKED/E_PYTHON_VERSION
PYTHON=python3 bash run.sh --mode preflight --execution-context target-board

# 仅在Linux ARM64、glibc>=2.28下离线安装到包内.runtime
# 使用系统Python3.8即可启动准备器；不sudo、不联网、不替换/usr/bin/python3
python3 -m experiments.devicekit.bootstrap

# run.sh自动优先使用包内.runtime/python/bin/python3；也可显式指定PYTHON
bash run.sh --mode preflight --execution-context target-board
bash run.sh --mode model-check --models small160 large224 --threads 2 --execution-context target-board
```

bootstrap预期PASS，并在报告中记录真实新Python/NumPy/Pillow/ORT/psutil版本和CPUExecutionProvider。已存在.runtime时拒绝覆盖；成功后直接用run.sh复测。失败时保留报告，重新试验使用新解压目录，不修改系统环境。

如果此前在v3遇到`../a/adm1178`的`FileNotFoundError`，用本修订完整包在新目录执行以上步骤即可；旧目录里的报告和失败的`.runtime-install-*`可保留。此前模型加载PASS来自已有环境，不能代替本次bootstrap复测。

新preflight预期PASS只是候选平台满足条件；dependencies等SKIP仍然是未执行。model-check预期两组model_load PASS，并记录实际CPU provider、线程、图接口，inference_executed=false；它仍未识别任何图片。若有E_DEPS/E_MODEL，报告含脱敏错误分类；详细日志留在本机private/，不要打包整个目录。

## 实际登记和单图

用操作者确认的真实照片、匿名sample-NNN、cat_id、真实session_id/event_id填写examples/enrollment.schema-example.jsonl。不要直接运行占位示例，不要把网络图标签伪装为现场真值。

```bash
bash run.sh --mode enroll --models small160 --enrollment /现场私有目录/enrollment.jsonl --execution-context target-board
bash run.sh --mode recognize --models small160 --gallery /登记结果/private/field-gallery.json --image /现场私有目录/image.jpg --execution-context target-board
bash run.sh --mode field --models small160 --enrollment /现场私有目录/enrollment.jsonl --field-queries /现场私有目录/queries.jsonl --execution-context target-board
```

图库、同名.calibration.json及.provenance.json一起保留。登记与查询场次独立，陌生校准/查询身份隔离；更换模型要重新登记校准。缺现场资料的enroll/field应BLOCKED，不能造真值。Large224需单独建匹配图库；默认Small只是速度参照。

本包没有历史149图/旧图库，historical/smoke缺件BLOCKED是预期。相机不自动测试，仅现场明确同意后使用--mode camera --camera INDEX；不操作猫门或GPIO。

## 只回传报告

每次命令打印Return only指向的ZIP及SHA；回传bootstrap、旧环境preflight、新环境preflight、model-check，以及实际完成的登记/识别/field报告。默认ZIP只含report.json/report.md/events.jsonl。不要回传private、照片、身份向量、完整安装目录。请另写设备型号和本次实际运行位置；--execution-context只是操作者声明。

详见MANUAL.zh.md、AGENT.md、docs/device-test-results.md。
