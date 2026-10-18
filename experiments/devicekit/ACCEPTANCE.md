# 板端测试边界

依次保存：旧解释器preflight的E_PYTHON_VERSION、bootstrap、新解释器preflight、两模型model-check、具备真实资料时的enroll/recognize/field。默认仅回传三文件报告ZIP及SHA。

包完整性PASS只证明封存字节。bootstrap PASS只证明包内安装与依赖导入。model-check PASS只证明真实CPU ONNX session和图接口；没有图片推理。现场DIR/FPIR需要真实独立场次、已确认身份和完整分母；不从历史图或x86速度外推。

最新ARM报告已确认修复后的包内离线安装、preflight及两模型CPU加载全部PASS；图片推理未执行。v4完整包包含冻结149图/图库/证据，用bash run-all.sh一次完成六组配置各三轮，汇总并回传一个ZIP。公开图指标不是现场验收，重复轮次不增加独立识别样本量；实际板端图像识别、持续性能及相机仍待验证。无数据的历史模式应BLOCKED，无相机请求应SKIP。不操作猫门/门锁/GPIO。
