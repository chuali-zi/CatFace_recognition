"""
门控业务层：对 Web/AI 提供 open/close/stop/reset/status。

[SAFETY-CRITICAL] 本文件包含电机通断、急停锁存、限位停止、定时动作完成。
[TIMED-EFFECTOR] 当前为定时效应器模式：没有限位时，motor_run_timeout_sec 到点即视为动作完成，
进入 open/closed，而不是 error；不需要 reset 恢复。限位仍可提前停止，属于可选增强。
[STATE-MACHINE] 状态变化通过 DoorStateMachine 判断，避免散落 if/else。
"""

import threading
import time

from safety import SafetyError
from state_machine import DoorState, DoorStateMachine


class DoorController(object):
    """门的本地安全 API；AI 不允许直接碰 GPIO，只能调用这里。"""

    def __init__(self, config, gpio, safety, logger):
        self.config = config
        self.gpio = gpio
        self.safety = safety
        self.logger = logger
        self.sm = DoorStateMachine()
        self.pins = config["gpio"]
        self.timing = config.get("timing", {})
        self._motor_enable_pins = self._resolve_motor_enable_pins()

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread = None
        self._motion_deadline = None

        self._setup_pins()
        with self._lock:
            self.state = self._infer_state_from_limits_locked()

    def start(self):
        """启动后台安全监控线程。"""
        self._thread = threading.Thread(target=self._monitor_loop, name="door-safety-monitor")
        self._thread.daemon = True
        self._thread.start()
        self.logger.info("DoorController started, state=%s", self.state)

    def open(self):
        """开门：已开不重复，反向运动前先停反向输出。"""
        return self._request_motion("open")

    def close(self):
        """关门：已关不重复，反向运动前先停反向输出。"""
        return self._request_motion("close")

    def stop(self):
        """普通停止：不断开急停锁存，只停止当前运动。"""
        with self._lock:
            if self.state != DoorState.EMERGENCY_STOP:
                self._stop_motors_locked(reason="manual_stop")
                if self.state in (DoorState.OPENING, DoorState.CLOSING):
                    self.state = DoorState.STOPPED
        return self.status()

    def reset(self):
        """复位：急停仍触发时禁止复位；否则清除 error/emergency 并按限位重建状态。"""
        with self._lock:
            if self.safety.emergency_stop_active():
                self.state = DoorState.EMERGENCY_STOP
                self._stop_motors_locked(reason="reset_blocked_by_estop")
                self.logger.warning("[SAFETY-CRITICAL] reset blocked: emergency stop still active")
                return self.status()

            self._stop_motors_locked(reason="reset")
            self.state = self._infer_state_from_limits_locked()
            self._motion_deadline = None
            self.logger.info("reset done, state=%s", self.state)
        return self.status()

    def status(self):
        """返回 Web/AI 可读取的安全状态快照。"""
        with self._lock:
            limits = self.safety.read_limits()
            return {
                "state": self.state,
                "limits": limits,
                "emergency_stop": self.safety.emergency_stop_active(),
                "backend": self.config.get("backend"),
                "ts": time.time(),
            }

    def shutdown(self):
        """程序退出：先停监控，再确保电机断电，最后清理 GPIO。"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        with self._lock:
            self._stop_motors_locked(reason="shutdown")
        self.gpio.cleanup()

    def _setup_pins(self):
        """[BOARD-CONFIG] 根据配置初始化输入/输出；输出默认全部关闭。"""
        output_pins = [
            int(self.pins["motor_open"]),
            int(self.pins["motor_close"]),
        ]
        output_pins.extend(self._motor_enable_pins)
        output_pins.append(int(self.pins["buzzer_status"]))
        for pin in self._unique_pins(output_pins):
            self.gpio.setup_output(pin, 0)

        active_low = bool(self.config.get("safety", {}).get("limit_switch_active_low", True))
        estop_active_low = bool(self.config.get("safety", {}).get("emergency_stop_active_low", True))
        self.gpio.setup_input(int(self.pins["limit_open"]), active_low=active_low)
        self.gpio.setup_input(int(self.pins["limit_close"]), active_low=active_low)
        self.gpio.setup_input(int(self.pins["emergency_stop"]), active_low=estop_active_low)

    def _request_motion(self, command):
        """[SAFETY-CRITICAL] 运动命令入口：急停/状态机/限位都通过才允许输出。"""
        with self._lock:
            if not self.sm.can_accept(self.state, command):
                self.logger.warning("[STATE-MACHINE] reject %s in state %s", command, self.state)
                return self.status()

            try:
                self.safety.guard_before_motion(command)
            except SafetyError as exc:
                self.logger.warning("[SAFETY-CRITICAL] motion blocked: %s", exc)
                if str(exc) == "emergency_stop_active":
                    self.state = DoorState.EMERGENCY_STOP
                    self._stop_motors_locked(reason="estop_before_motion")
                return self.status()

            # [SAFETY-CRITICAL] 换向前先停掉另一方向，避免 H 桥/继电器直通。
            self._stop_motors_locked(reason="before_%s" % command)
            time.sleep(float(self.timing.get("relay_settle_sec", 0.2)))

            if command == "open":
                self.gpio.write(int(self.pins["motor_open"]), 1)
                self.gpio.write(int(self.pins["motor_close"]), 0)
            else:
                self.gpio.write(int(self.pins["motor_open"]), 0)
                self.gpio.write(int(self.pins["motor_close"]), 1)

            self._write_motor_enable_locked(1)
            self.gpio.write(int(self.pins["buzzer_status"]), 1)
            self.state = self.sm.moving_state_for(command)
            self._motion_deadline = time.time() + self.safety.motion_timeout_sec()
            self.logger.info("motion started command=%s state=%s", command, self.state)
        return self.status()

    def _monitor_loop(self):
        """[SAFETY-CRITICAL] 后台循环：急停反馈 > 限位到位 > 定时完成；硬件急停断电优先于软件。"""
        poll = float(self.timing.get("poll_interval_sec", 0.1))
        while not self._stop_event.is_set():
            with self._lock:
                if self.safety.emergency_stop_active():
                    if self.state != DoorState.EMERGENCY_STOP:
                        self.logger.error("[SAFETY-CRITICAL] emergency stop triggered")
                    # [SAFETY-CRITICAL] 软件只补停和锁存；硬件急停必须已独立切断电机/驱动器电源或使能。
                    self.state = DoorState.EMERGENCY_STOP
                    self._stop_motors_locked(reason="emergency_stop")
                else:
                    limits = self.safety.read_limits()
                    finished = None
                    if self.state == DoorState.OPENING and limits["open"]:
                        finished = self.sm.limit_finished_state(self.state, "open")
                    elif self.state == DoorState.CLOSING and limits["close"]:
                        finished = self.sm.limit_finished_state(self.state, "close")

                    if finished:
                        self._stop_motors_locked(reason="limit_reached")
                        self.state = finished
                        self._motion_deadline = None
                        self.logger.info("[STATE-MACHINE] limit reached, state=%s", self.state)
                    elif self.state in (DoorState.OPENING, DoorState.CLOSING):
                        if self._motion_deadline and time.time() > self._motion_deadline:
                            # [TIMED-EFFECTOR] 运行时间到达 motor_run_timeout_sec：视为动作完成，
                            # 停止电机并进入 open/closed；不进入 error，不需要 reset。
                            self._stop_motors_locked(reason="motion_timer_done")
                            if self.state == DoorState.OPENING:
                                self.state = DoorState.OPEN
                            else:
                                self.state = DoorState.CLOSED
                            self._motion_deadline = None
                            self.logger.info("[TIMED-EFFECTOR] motion time done, state=%s", self.state)
            time.sleep(poll)

    def _stop_motors_locked(self, reason):
        """[SAFETY-CRITICAL] 软件统一断电顺序：先断使能，再断方向；不替代硬件急停断电链路。"""
        self._write_motor_enable_locked(0)
        self.gpio.write(int(self.pins["motor_open"]), 0)
        self.gpio.write(int(self.pins["motor_close"]), 0)
        self.gpio.write(int(self.pins["buzzer_status"]), 0)
        self.logger.info("motors stopped reason=%s", reason)

    def _write_motor_enable_locked(self, value):
        """[BOARD-CONFIG] 支持 IBT-2/BTS7960 的 R_EN/L_EN 分别接线。"""
        for pin in self._motor_enable_pins:
            self.gpio.write(pin, value)

    def _resolve_motor_enable_pins(self):
        """返回电机使能 GPIO；新配置优先使用 R_EN/L_EN，旧配置回退到 motor_enable。"""
        if "motor_enable_r" in self.pins or "motor_enable_l" in self.pins:
            pins = []
            if "motor_enable_r" in self.pins:
                pins.append(int(self.pins["motor_enable_r"]))
            if "motor_enable_l" in self.pins:
                pins.append(int(self.pins["motor_enable_l"]))
            return self._unique_pins(pins)
        return (int(self.pins["motor_enable"]),)

    @staticmethod
    def _unique_pins(pins):
        unique = []
        for pin in pins:
            if pin not in unique:
                unique.append(pin)
        return tuple(unique)

    def _infer_state_from_limits_locked(self):
        """根据限位推断上电初始状态；两个限位都异常时进入 stopped/error 需谨慎。"""
        limits = self.safety.read_limits()
        if limits["open"] and not limits["close"]:
            return DoorState.OPEN
        if limits["close"] and not limits["open"]:
            return DoorState.CLOSED
        if limits["open"] and limits["close"]:
            self.logger.error("[SAFETY-CRITICAL] both limits active, check wiring")
            return DoorState.ERROR
        return DoorState.STOPPED
