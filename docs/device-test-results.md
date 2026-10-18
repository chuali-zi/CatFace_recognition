# 设备结果协议 v3

每次使用新的 run_id/目录，默认回传 ZIP 仅含 `report.json`、`report.md`、`events.jsonl`，同目录另有 `.zip.sha256`。ZIP文件名、SHA256、JSON的run_id应对应；events按顺序等于JSON的stages，Markdown展示相同状态。原始日志/照片/图库/向量留在private或原始位置，不默认收集。

`schema_version=3`兼容既有stage/status/code字段。status仅PASS、FAIL、BLOCKED、SKIP；overall优先FAIL，其次BLOCKED，否则PASS。程序退出码依次1、2、0。PASS仅代表请求的诊断完成，SKIP表示未执行。总状态不能替代逐阶段状态。

普通测试报告保留：mode、run_id、开始结束时间、真实environment、源码/包清单/资产/模型SHA256、commands、scope、stages和计时规则。`operator_execution_context=target-board`只是操作者声明；`hardware_validated=false`、`recognition_validated=false`不能自行改成true。硬件型号由操作者另行记录，架构/phytium内核提供环境证据但不证明全部硬件路径通过。

v3新增：

| 字段 | 含义 |
|---|---|
| platform_assessment | compatible及逐项findings，含component/actual/required/code/remediation；Python3.8明确E_PYTHON_VERSION，libc等另报。 |
| dependency_inventory | 每个核心包实际安装元数据/固定版本、MISSING/PIN_MATCH/PIN_MISMATCH；import_tested=false，不冒充成功导入。 |
| recovery_command | 匹配Ubuntu20.04 ARM64的包内离线bootstrap入口；不替换系统Python。 |
| dependencies阶段error_summary | 仅白名单异常类型、缺失模块名、动态库/CPU指令类别，不导出原始日志或私有路径；exit_code独立记录。 |
| model_load阶段evidence | 实际CPU session providers、线程、输入尺寸/特征维度、图库绑定情况；inference_executed=false。 |

bootstrap报告同为schema3，但mode=bootstrap，记录系统解释器和真实installed_environment、network_used=false、system_python_modified=false、model_inference_executed=false。没有模型/现场分数。不支持的架构、缺依赖、校验失败或既有.runtime均BLOCKED，保留本机失败日志和脱敏的报告；失败可能留下独立临时目录，使用新解压目录重试，不覆盖已有环境。

2026-10-18原报告为schema2，仍可按原阶段解析。其preflight BLOCKED不能改写成模型失败；7项SKIP无模型/性能信息。新报告需同时留存原ZIP、报告ZIP SHA、设备型号与每次实际命令。model-check不能产出DIR/FPIR，单帧不能产出有意义的P50/P95；缺真实身份/独立场次时不能编造现场指标。

有网主机可重准备运行时：`python -m experiments.devicekit.prepare_runtime --out 新目录`。它下载固定官方运行时与11项目标wheel，按官方SHA/PyPI元数据核验后生成锁文件；不会执行ARM程序。完整交付使用`experiments.devicekit.release`，要求已核验的旧包、封存运行时及新目录，按白名单复制，不使用整个工作树或旧验收结果充当新包验收。

## v4一次运行的汇总包

`bash run-all.sh`使用suite入口，单项诊断仍沿用schema3。suite的`report.json`为schema1、mode=suite，描述安装、历史完整矩阵、重复/线程一致性以及可选field/camera的状态；不与单项schema3混读。

默认两模型、1/2/4线程、三轮，每组149个独立查询，计时共2682次（不含首帧/预热）。DIR/FPIR只取第一轮，失败仍在分母。`--limit`或自选模型/线程用于排错，参数及limited_run保留，不能宣称默认六组已完成。

唯一回传ZIP含suite的report.json/report.md、summary.json/summary.csv、observations.jsonl和实际执行的各项脱敏诊断JSON（安装时另附bootstrap.json）。匿名观测仅含稳定sample-NNN/cat-NNN、角色/状态/分数/计时/轮次，不含图片路径、原始身份标签、照片、图库或原始日志。summary保留每轮P50/P95、传感器首末/极值、CPU provider、首帧、内存和同模型跨线程/重复决策变化。传感器不可用为空字典。

private留在设备，包含原始测量、遥测和日志；ZIP不遍历该目录。某配置失败仍继续其他配置并回传部分结果。套件PASS只表示请求的测量完成；一致性变化BLOCKED需要核查数值/阈值，既有开发集表现不等于现场效果。
