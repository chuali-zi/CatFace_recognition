# 初学者临时学习清单

## Linux GPIO sysfs
- `/sys/class/gpio/export`：写 GPIO 编号，导出引脚。
- `/sys/class/gpio/gpioN/direction`：设置 `in` 或 `out`。
- `/sys/class/gpio/gpioN/value`：读写 0/1。
- `active_low`：有些输入低电平表示有效，程序里要统一成“是否触发”。

## 状态机
- 状态：当前门处于什么阶段，例如 `closed/opening/open`。
- 事件：按钮、限位、急停、超时。
- 原则：同一时刻只能有一个主要状态；危险事件优先级最高。

## 电机正反转与互锁
- 开门和关门方向不能同时输出，否则可能短路或损坏驱动器。
- 换向前先停使能，再等继电器释放，最后才给另一方向上电。

## Web API
- `GET /api/status`：查询状态。
- `POST /api/open` 等：发送动作命令。
- 前端只是“遥控器”，真正安全判断必须在 Python 后端完成。

## 硬件急停与负载驱动
- 急停必须硬件级切断电机/驱动器电源或使能；GPIO510 只是急停状态反馈。
- 软件急停只是第二层保护，因为程序可能卡死、系统可能崩、GPIO 可能失控。
- IO 不能直接接电机、继电器线圈、电磁锁；必须经过驱动模块、光耦、MOS、继电器模块或电机驱动板。

## 调试串口 J2
- 参数：115200，8N1。
- USB-TTL 要 TX/RX 交叉、GND 共地。
- 不要用 5V TTL 直连；接线前确认 DEBUG_UART 实际工作电平，必要时加电平转换模块。
- J2 有 1.8V IO 提示，不要默认所有信号都是 3.3V。

## AI 安全边界
- AI 只能输出白名单命令：open/close/stop/reset。
- 模型 key 不放代码里，用环境变量。
- 即使接了大模型，也必须经过 `_normalize_command` 和 DoorController。
