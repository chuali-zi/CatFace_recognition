"""
门状态机模块。

[STATE-MACHINE] 只负责“状态含义”和“命令是否允许”，不直接碰 GPIO。
真正断电、停电机、急停锁存在 DoorController/SafetySupervisor 中执行。
"""


class DoorState(object):
    CLOSED = "closed"
    OPENING = "opening"
    OPEN = "open"
    CLOSING = "closing"
    STOPPED = "stopped"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"


ALL_STATES = (
    DoorState.CLOSED,
    DoorState.OPENING,
    DoorState.OPEN,
    DoorState.CLOSING,
    DoorState.STOPPED,
    DoorState.ERROR,
    DoorState.EMERGENCY_STOP,
)


class DoorStateMachine(object):
    """门状态判断工具：保持简单，方便初学者直接看规则。"""

    def can_accept(self, state, command):
        """[STATE-MACHINE] 判断当前状态下是否接受 open/close/stop/reset。"""
        if command == "reset":
            return True
        if state == DoorState.EMERGENCY_STOP:
            return False
        if state == DoorState.ERROR:
            return command == "stop"
        if command == "stop":
            return True
        # [STATE-MACHINE] 已开/已关的重复动作在状态机层拒绝，限位保护只是第二层安全保护。
        if command == "open":
            return state in (DoorState.CLOSED, DoorState.STOPPED, DoorState.CLOSING)
        if command == "close":
            return state in (DoorState.OPEN, DoorState.STOPPED, DoorState.OPENING)
        return False

    def moving_state_for(self, command):
        """[STATE-MACHINE] 把运动命令映射成运动状态。"""
        if command == "open":
            return DoorState.OPENING
        if command == "close":
            return DoorState.CLOSING
        raise ValueError("unsupported motion command: %s" % command)

    def limit_finished_state(self, state, limit_name):
        """[STATE-MACHINE] 运动中碰到限位后应该进入哪个稳定状态。"""
        if state == DoorState.OPENING and limit_name == "open":
            return DoorState.OPEN
        if state == DoorState.CLOSING and limit_name == "close":
            return DoorState.CLOSED
        return None
