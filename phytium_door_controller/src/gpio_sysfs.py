"""
飞腾派真实 GPIO 后端：通过 Linux sysfs 操作 GPIO。

[GPIO-SYSFS] 目标系统按开发手册使用 /sys/class/gpio。
[BOARD-CONFIG] GPIO 编号来自配置文件，默认使用 gpiochip 基地址+偏移后的全局编号。
[HARDWARE-TODO] 上板后必须确认：引脚是否被复用、继电器高低有效、限位/急停电平。
"""

import os
import time

from gpio_base import GPIOBackend


class SysfsGPIO(GPIOBackend):
    """最小 sysfs GPIO 封装；只实现本项目需要的输入/输出。"""

    SYSFS_ROOT = "/sys/class/gpio"

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.safety_cfg = config.get("safety", {})
        self._exported = set()
        self._output_pins = set()
        self._owned_exports = set()
        self._input_active_low = {}

    def setup_output(self, pin, initial=0):
        pin = int(pin)
        self._export(pin)
        self._write_file(self._pin_path(pin, "direction"), "out")
        self._output_pins.add(pin)
        self.write(pin, initial)

    def setup_input(self, pin, active_low=True):
        pin = int(pin)
        self._export(pin)
        self._write_file(self._pin_path(pin, "direction"), "in")
        self._input_active_low[pin] = bool(active_low)

    def write(self, pin, value):
        """[GPIO-SYSFS] 写 value 文件；输出有效电平可在配置中调整。"""
        logical = bool(value)
        raw = self._logical_to_raw(pin, logical)
        self._write_file(self._pin_path(pin, "value"), "1" if raw else "0")

    def read(self, pin):
        """[GPIO-SYSFS] 读 value 文件，并按 active_low 转成“是否触发”。"""
        text = self._read_file(self._pin_path(pin, "value")).strip()
        raw = text == "1"
        active_low = self._input_active_low.get(pin, True)
        return (not raw) if active_low else raw

    def cleanup(self):
        # [SAFETY-CRITICAL] 关闭输出必须走 write(pin, 0) 做“逻辑关闭”，
        # 由 _logical_to_raw 处理低电平有效设备，避免 raw 0 被误当成关闭电平。
        for pin in list(self._output_pins):
            try:
                self.write(pin, 0)
            except Exception as exc:  # pragma: no cover - 清理阶段只记录
                self.logger.warning("[GPIO-SYSFS] 输出关闭失败 pin=%s err=%s", pin, exc)

        # [GPIO-SYSFS] 只 unexport 本程序 export 过的 GPIO，避免影响系统里已存在的 GPIO。
        for pin in list(self._owned_exports):
            try:
                self._write_file(os.path.join(self.SYSFS_ROOT, "unexport"), str(pin))
            except Exception as exc:  # pragma: no cover
                self.logger.warning("[GPIO-SYSFS] unexport 失败 pin=%s err=%s", pin, exc)

        self._output_pins.clear()
        self._owned_exports.clear()
        self._exported.clear()

    def _export(self, pin):
        pin = int(pin)
        gpio_dir = os.path.join(self.SYSFS_ROOT, "gpio%s" % pin)
        if os.path.exists(gpio_dir):
            # [GPIO-SYSFS] 已存在的 GPIO 可能是系统/其他程序导出，只能使用，不能 unexport。
            self._exported.add(pin)
            return

        self._write_file(os.path.join(self.SYSFS_ROOT, "export"), str(pin))
        # [GPIO-SYSFS] export 后内核创建 gpioN 目录可能有短暂延迟。
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if os.path.exists(gpio_dir):
                self._owned_exports.add(pin)
                self._exported.add(pin)
                return
            time.sleep(0.05)
        raise RuntimeError("[GPIO-SYSFS] export timeout pin=%s" % pin)

    def _pin_path(self, pin, name):
        return os.path.join(self.SYSFS_ROOT, "gpio%s" % pin, name)

    def _logical_to_raw(self, pin, logical):
        """把业务逻辑电平转换为实际写进 sysfs 的 0/1。"""
        gpio = self.config.get("gpio", {})
        motor_enable_pins = set()
        if "motor_enable" in gpio:
            motor_enable_pins.add(int(gpio["motor_enable"]))
        if "motor_enable_r" in gpio:
            motor_enable_pins.add(int(gpio["motor_enable_r"]))
        if "motor_enable_l" in gpio:
            motor_enable_pins.add(int(gpio["motor_enable_l"]))
        if pin in motor_enable_pins:
            active_high = bool(self.safety_cfg.get("motor_enable_active_high", True))
        elif pin == int(gpio.get("buzzer_status", -2)):
            active_high = bool(self.safety_cfg.get("buzzer_active_high", True))
        else:
            active_high = bool(self.safety_cfg.get("motor_direction_active_high", True))
        return logical if active_high else (not logical)

    @staticmethod
    def _write_file(path, data):
        with open(path, "w") as f:
            f.write(data)

    @staticmethod
    def _read_file(path):
        with open(path, "r") as f:
            return f.read()
