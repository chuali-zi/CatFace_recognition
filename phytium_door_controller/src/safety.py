"""
安全保护模块。

[SAFETY-CRITICAL] 急停、限位集中在这里做“读取和判断”。
[TIMED-EFFECTOR] motor_run_timeout_sec 不再是故障超时，而是定时效应器模式下的动作运行时长。
电机真正断电由 DoorController.stop_motors 完成，避免安全逻辑分散。
[SAFETY-CRITICAL] GPIO510 只是急停状态反馈；真正的电机/驱动器断电必须由硬件急停回路独立完成。
"""


class SafetyError(Exception):
    """安全条件不满足时抛出，业务层必须捕获并阻止动作。"""


class SafetySupervisor(object):
    """读取急停/限位，给出运动前安全检查。"""

    def __init__(self, config, gpio, logger):
        self.config = config
        self.gpio = gpio
        self.logger = logger
        self.pins = config["gpio"]
        self.timing = config.get("timing", {})

    def emergency_stop_active(self):
        """[SAFETY-CRITICAL] 读取急停反馈；程序卡死/系统崩溃/GPIO 失控时，硬件急停仍必须能断电。"""
        return bool(self.gpio.read(int(self.pins["emergency_stop"])))

    def read_limits(self):
        """[SAFETY-CRITICAL] 返回开/关限位是否触发。"""
        return {
            "open": bool(self.gpio.read(int(self.pins["limit_open"]))),
            "close": bool(self.gpio.read(int(self.pins["limit_close"]))),
        }

    def guard_before_motion(self, command):
        """[SAFETY-CRITICAL] 运动前检查：急停优先，已到目标限位则不重复动作。"""
        if self.emergency_stop_active():
            raise SafetyError("emergency_stop_active")

        limits = self.read_limits()
        if command == "open" and limits["open"]:
            raise SafetyError("already_at_open_limit")
        if command == "close" and limits["close"]:
            raise SafetyError("already_at_close_limit")
        return limits

    def motion_timeout_sec(self):
        """[TIMED-EFFECTOR] 开门/关门动作的运行时长；到点视为动作完成（无硬件限位时的定时完成时间）。"""
        return float(self.timing.get("motor_run_timeout_sec", 12.0))
