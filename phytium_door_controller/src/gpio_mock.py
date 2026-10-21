"""
模拟 GPIO 后端：用于 Windows/普通电脑无硬件开发。

[BOARD-CONFIG] 引脚编号从配置文件读取；这里不直接对应真实板卡电平。
[HARDWARE-TODO] 上飞腾派后请把 backend 改为 sysfs，并核对真实限位/急停电平。
"""

import threading
import time

from gpio_base import GPIOBackend


class MockGPIO(GPIOBackend):
    """用内存字典模拟 GPIO 输入/输出，便于先把软件流程跑通。"""

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.mock_cfg = config.get("mock", {})
        self.simulate_limits = bool(self.mock_cfg.get("simulate_limits", True))
        self.travel_sec = float(self.mock_cfg.get("mock_travel_sec", 3.0))

        self._outputs = {}
        self._inputs = {}
        self._input_active_low = {}
        self._lock = threading.Lock()
        self._motion_dir = None
        self._motion_start = None

    def setup_output(self, pin, initial=0):
        with self._lock:
            self._outputs[pin] = 1 if initial else 0
        self.logger.info("[MOCK-GPIO] setup_output pin=%s initial=%s", pin, initial)

    def setup_input(self, pin, active_low=True):
        with self._lock:
            self._inputs[pin] = False
            self._input_active_low[pin] = active_low
        self.logger.info("[MOCK-GPIO] setup_input pin=%s active_low=%s", pin, active_low)

    def write(self, pin, value):
        with self._lock:
            self._outputs[pin] = 1 if value else 0
            self._refresh_motion_locked()
        self.logger.info("[MOCK-GPIO] write pin=%s value=%s", pin, 1 if value else 0)

    def read(self, pin):
        with self._lock:
            self._simulate_limits_locked()
            return bool(self._inputs.get(pin, False))

    def set_mock_input(self, pin, active):
        """测试辅助：手动设置某个输入是否触发。"""
        with self._lock:
            self._inputs[pin] = bool(active)

    def cleanup(self):
        self.logger.info("[MOCK-GPIO] cleanup done")

    def _pin(self, name):
        return int(self.config["gpio"][name])

    def _motor_enable_pins(self):
        gpio = self.config["gpio"]
        if "motor_enable_r" in gpio or "motor_enable_l" in gpio:
            pins = []
            if "motor_enable_r" in gpio:
                pins.append(int(gpio["motor_enable_r"]))
            if "motor_enable_l" in gpio:
                pins.append(int(gpio["motor_enable_l"]))
            return tuple(pins)
        return (self._pin("motor_enable"),)

    def _is_on(self, pin):
        return bool(self._outputs.get(pin, 0))

    def _refresh_motion_locked(self):
        """根据当前输出判断模拟门应该往哪个方向运动。"""
        motor_open = self._pin("motor_open")
        motor_close = self._pin("motor_close")
        motor_enable_pins = self._motor_enable_pins()

        if not all(self._is_on(pin) for pin in motor_enable_pins):
            new_dir = None
        elif self._is_on(motor_open) and not self._is_on(motor_close):
            new_dir = "open"
        elif self._is_on(motor_close) and not self._is_on(motor_open):
            new_dir = "close"
        else:
            new_dir = None

        if new_dir != self._motion_dir:
            self._motion_dir = new_dir
            self._motion_start = time.time() if new_dir else None

    def _simulate_limits_locked(self):
        """模拟门运动一段时间后触发限位，让模拟模式可以完整跑流程。"""
        if not self.simulate_limits or not self._motion_dir or not self._motion_start:
            return

        if time.time() - self._motion_start < self.travel_sec:
            return

        limit_open = self._pin("limit_open")
        limit_close = self._pin("limit_close")
        if self._motion_dir == "open":
            self._inputs[limit_open] = True
            self._inputs[limit_close] = False
        elif self._motion_dir == "close":
            self._inputs[limit_open] = False
            self._inputs[limit_close] = True

        self._motion_dir = None
        self._motion_start = None
