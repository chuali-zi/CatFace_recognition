# 测试Agent指南

先读README、QUICKSTART、MANUAL与LICENSE_REVIEW。只做猫个体识别研究，无门锁/电机/GPIO/UART/通行权限动作。

1. v4在授权Linux设备新解压目录执行bash run-all.sh，一次完成安装检查和完整图片/性能矩阵；单项排错用bash run.sh。自检PASS不是识别PASS。
2. model-check需要真模型和CPU依赖，不需要照片/图库；只报告实际加载/图接口。
3. enroll/recognize/field按资源表要求实际数据、匹配图库及校准证据。来源/许可记录和SHA必查，distribution_approved是广义分发声明，不阻止已有授权的本地科研计算，也不是法律许可证明。
4. 只有实际操作者确认身份和真实session/event才填identity_verified=true。发布者网络图标签不能冒充现场身份/来访真值。
5. 登记图库、校准证据和来源回执一起保留；不能从历史阈值编造现场拒识。field登记/查询场次和陌生身份隔离。
6. historical/smoke要求完整149图、冻结协议/图库/证据；v4完整包包含，v3模型包不含。缺件诚实BLOCKED，不能生成假分数。
7. --camera仅现场明确同意时执行，一帧后丢弃；不sudo/改组/改驱动/绕过设备权限。
8. 只回传终端Return only指向的ZIP和SHA。suite默认包含匿名观测/六组汇总和单项诊断；单项模式仍是三文件ZIP。private/照片/原始猫标签/登记向量不默认收集，不zip整个结果/资产目录。
9. 写明真实设备/软件、mode、run_id及每阶段PASS/FAIL/BLOCKED/SKIP。加载、研究图、少量smoke、本机x86_64都不能写成ARM/相机/现场验收。

10. 系统Python3.8仅用于启动标准库准备器；v4入口会自动选择可用包内环境或安装.runtime。新版ARM离线安装已获板端PASS，不替换系统Python、不sudo、不改libc。不要把准备器PASS当识别PASS。
11. 报告schema3保留stage/status/code，新增platform_assessment、dependency_inventory及模型加载evidence；准确回传实际模式和三文件ZIP/SHA。error_summary仅含白名单错误分类，详细日志留本地。
