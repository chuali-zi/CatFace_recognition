# v3复测边界

依次保存：旧解释器preflight的E_PYTHON_VERSION、bootstrap、新解释器preflight、两模型model-check、具备真实资料时的enroll/recognize/field。默认仅回传三文件报告ZIP及SHA。

包完整性PASS只证明封存字节。bootstrap PASS只证明包内安装与依赖导入。model-check PASS只证明真实CPU ONNX session和图接口；没有图片推理。现场DIR/FPIR需要真实独立场次、已确认身份和完整分母；不从历史图或x86速度外推。

当前ARM报告仅旧版preflight BLOCKED；v3 ARM运行时、CPU指令/动态库、实际图像识别、持续性能及相机仍需目标设备执行。无数据的历史模式应BLOCKED，无相机请求应SKIP。不操作猫门/门锁/GPIO。
