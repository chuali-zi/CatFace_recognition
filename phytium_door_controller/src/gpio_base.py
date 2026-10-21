"""
GPIO 抽象接口层。

作用：把“门控业务”和“具体 GPIO 实现”分开。
- 普通电脑上使用 MockGPIO 模拟运行。
- 飞腾派 CEK8903 上使用 SysfsGPIO 操作 /sys/class/gpio。

[BOARD-CONFIG] 所有引脚编号都必须来自配置文件，不要在这里写死。
"""


class GPIOBackend(object):
    """GPIO 后端最小接口：业务层只依赖这些方法。"""

    def setup_output(self, pin, initial=0):
        """把某个 GPIO 配置为输出，并给初始电平。"""
        raise NotImplementedError

    def setup_input(self, pin, active_low=True):
        """把某个 GPIO 配置为输入。active_low 表示低电平是否算“触发”。"""
        raise NotImplementedError

    def write(self, pin, value):
        """输出逻辑值：1/0 或 True/False。"""
        raise NotImplementedError

    def read(self, pin):
        """读取输入，返回 True 表示“触发/有效”，False 表示未触发。"""
        raise NotImplementedError

    def cleanup(self):
        """释放 GPIO 资源。程序退出时必须调用。"""
        raise NotImplementedError
