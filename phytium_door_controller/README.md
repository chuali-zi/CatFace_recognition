# 飞腾派 CEK8903/PIQD2/4 机械门控制原型

## 仓库同步说明（2026-10-21）

按项目负责人的要求，本目录作为独立门控原型同步到 CatFace_recognition 仓库，单独启动；不接入或改变 `src/catid/` 的识别逻辑，不根据识别结果自动开门。本次保留现有代码、GPIO 配置和 12 秒动作时长，不代表重新完成板端或硬件安全验收。

**接线文档尚待复核：** `docs/pin_map.md` 等旧文档将 gpio450 标为 J1-18，而此前用户反馈实测对应 J1-16。此次不据此改动代码或配置；不能把旧表当作已确认的实物接线指南，重新接线前应断电并核对实际板卡。

默认 `mock` 后端仅供无硬件模拟。未配置控制令牌时不要将控制服务暴露到不可信网络。定时到点后的 `open`/`closed` 是软件推定，不是位置传感器确认；硬件急停保护仍不可由软件替代。

目标：在萤火工场飞腾派 CEK8903-PIQD2/4 上控制小型机械门；先在普通电脑用 `MockGPIO` 跑通，再上板切换 `SysfsGPIO` 操作 `/sys/class/gpio`。

## 1. 目录结构
- `config/door_config.example.json`：全部可配置项（GPIO、后端、超时、Web、AI）。
- `src/`：Python 后端代码，按模块拆分。
- `static/`：纯 HTML/CSS/JavaScript 控制页。
- `scripts/`：Windows/Linux 模拟启动脚本、上板启动脚本。
- `docs/`：引脚表、接线规划、上板核对表、初学者学习点。

## 2. 启动模拟版（Windows/普通电脑）
要求：Python 3.8+，只用标准库。

方式一：双击/执行脚本
```powershell
cd phytium_door_controller
.\scripts\run_mock.ps1
```

方式二：手动启动
```powershell
cd phytium_door_controller
$env:DOOR_BACKEND="mock"
python src/main.py
```

启动后访问：`http://127.0.0.1:8080/`

模拟模式默认开启 `mock.simulate_limits=true`，开门/关门约 3 秒后自动触发限位，方便无硬件验证状态机。

## 3. Web API
- `GET /api/status`
- `POST /api/open`
- `POST /api/close`
- `POST /api/stop`
- `POST /api/reset`
- `POST /api/ai-command`，body 示例：`{"text":"请开门"}`

控制接口 token：
- 配置 `web.control_token` 默认为空：所有 `POST /api/*` 不鉴权，只适合本机或实验环境。
- `web.control_token` 非空时：所有 `POST /api/*` 必须带请求头 `X-Door-Token: <token>`，否则返回 403。
- 也可用环境变量 `DOOR_WEB_TOKEN` 覆盖；不要把 token 写进前端代码或提交到仓库。
- 前端会从浏览器 `localStorage.getItem("door_token")` 读取 token 并自动带到 POST 请求头。
- `GET /api/status` 当前暂不强制 token；正式产品如果暴露到局域网，也应考虑限制状态接口，因为状态和日志可能泄露设备信息。

## 4. 修改 GPIO 编号/端口/超时
只改 `config/door_config.example.json`，不要改代码里的引脚。

常用项：
- `gpio.*`：门控引脚全局编号。
- `backend`：`mock` 或 `sysfs`。
- `timing.motor_run_timeout_sec`：开门/关门动作的运行时长。当前为“定时效应器”模式：该时间既是最大运行时间，也是无硬件限位时的定时动作完成时间——到点自动停止电机并进入 `open`/`closed`，不会进入 `error`，也不需要 `reset`。如果接了限位开关，限位触发仍会提前停止并完成动作。
- `timing.relay_settle_sec`：换向前继电器释放等待；继电器先用 0.2s-0.5s 更稳，MOS/H 桥确认后才考虑缩短。
- `web.host/web.port`：Web 服务监听地址。
- `safety.*_active_low/active_high`：输入/输出有效电平。

也支持环境变量覆盖：
- `DOOR_CONFIG`：配置文件路径
- `DOOR_BACKEND`：`mock/sysfs`
- `DOOR_WEB_HOST`、`DOOR_WEB_PORT`、`DOOR_WEB_TOKEN`
- `DOOR_AI_ENABLED`、`DOOR_AI_PROVIDER`
- `KIMI_API_KEY`、`GLM_API_KEY`、`OPENAI_COMPATIBLE_BASE_URL`

## 5. 部署到飞腾派并切换真实 GPIO
1. 把整个 `phytium_door_controller/` 目录复制到板子。
2. 确认系统为 Ubuntu 20.04 arm64、Python 3.8.10。
3. 按 `docs/pin_map.md`、`docs/wiring_plan.md` 和 `docs/hardware_bringup_checklist.md` 接线、测量并复核。
4. 将配置改为真实后端：
   ```json
   "backend": "sysfs"
   ```
   或启动时设置：
   ```bash
   export DOOR_BACKEND=sysfs
   ```
5. 启动：
   ```bash
   chmod +x scripts/run_on_board.sh
   ./scripts/run_on_board.sh
   ```

`[GPIO-SYSFS]` 真实后端会写 `/sys/class/gpio/export`、`direction`、`value`；通常需要 root 权限或配置 udev/用户组。

## 6. 上板前必须确认的硬件事项
- 引脚是否被 I2C1/UART2/I2C8/SPI0/I2S/MIO 默认功能占用；当前配置已避开这些脚。
- 限位开关、急停输入到底是高电平触发还是低电平触发。
- 继电器/电机驱动器是高电平有效还是低电平有效。
- 电机方向 A/B 是否互锁，是否必须先断使能再换向。
- 开门/关门方向如果与机械实际相反，优先改配置映射或驱动器接线，不要改状态机；状态机保持抽象 `open/close`。
- 急停必须硬件级切断电机/驱动器电源或使能，这不是可选项；软件急停只是第二层保护，因为程序可能卡死、系统可能崩、GPIO 可能失控。
- 正确急停结构：急停按钮一边切断电机/驱动器电源或使能，一边给飞腾派 GPIO510 状态反馈。
- 调试串口 J2 DEBUG_UART：115200，8N1；USB-TTL 必须 TX/RX 交叉、GND 共地。
- 不要用 5V TTL 直连 J2；接线前必须确认 DEBUG_UART 实际工作电平，电平不匹配时用电平转换模块。
- J2 上存在 1.8V IO 提示，不要凭经验默认所有 J2 信号都是 3.3V。

## 7. 为什么不能 IO 直接接电机/继电器/电磁锁
开发板 IO 电流小、耐压低，不能直接驱动感性负载；电机和继电器线圈断电会产生反向高压，容易损坏 IO。必须使用光耦隔离、继电器/接触器、电机驱动器、续流二极管/TVS 等外围电路，并保留独立急停断电链路。

## 8. 可选依赖说明
当前版本只使用 Python 标准库。后续如果 Web API 变复杂，可在独立分支评估 FastAPI/Flask；但需要在 README 明确新增依赖、安装命令和对 Python 3.8 的兼容性。前端保持普通 HTML/CSS/JavaScript，不引入 React/Vue。
