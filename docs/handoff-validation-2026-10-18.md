# 2026-10-18 交接验证

源码、文档和脱敏板端报告准备同步到现有GitHub项目；运行时/模型/图片留在独立本地交接包。以下区分测试主机与真实飞腾派。

## 仓库检查

`python -m pytest -q --basetemp experiments/runs/handoff-release-pytest`：122 passed，1 skipped。跳过项为Windows无法实际创建符号链接的用例；无显式目录的符号链接父目录回归仍已通过。Python源码编译和`git diff --check`通过。

新增的必要回归覆盖匿名导出不含路径/原始身份、跨线程决策变化、传感器缺失、错误查询保留分母、完整入口异常时仍打包脱敏结果，以及系统Python3.8可以解析suite。

## 真实图片完整矩阵

测试主机为同一工作站的WSL / AlmaLinux9 x86_64，CPython3.11.13，NumPy1.26.4、Pillow11.1.0、ORT1.20.1、psutil6.1.1。不是ARM设备，不把其耗时外推飞腾派。

从生成的v4目录实际执行`bash run-all.sh --execution-context unknown`。Small160/Large224各1/2/4线程，每组149图×3轮，另有首帧和预热：

- 六组全部完成，共2682次计时查询，运行错误0。
- 每组447个总耗时观测，准确率仍只用首轮29张已知/120张陌生。
- Small160每组正确接受19/29、陌生误接受0/120；Large224每组24/29、0/120。这是旧development池的固定单图库重放，不替代重复图库/新身份池的已有负面结果，也不是新的独立验证。
- 同一模型跨线程、跨重复轮次的最终识别决策变化0，本次最大分数差0。
- suite报告PASS，field/camera因未请求为SKIP；单一汇总ZIP生成成功。

本地完整记录在`experiments/runs/handoff-v4-full-replay/`及同名ZIP。最后交付包将从提交后的源码重建，再验证其入口、包清单及脱敏回传结构；这项检查结果见交接包旁`交接校验.json`。

## 飞腾派已验证范围

见[最新板端JSON](results/board-2026-10-18/index.json)：ARM包内离线安装、preflight及两组模型CPU加载均PASS；尚无板端图片重放/时延/相机结果。最新source_commit为23f7cbe，不能把本页WSL完整矩阵当成该设备已完成图片测试。
