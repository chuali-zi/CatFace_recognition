"""
程序入口：加载配置、创建 GPIO 后端、启动门控安全线程和 Web 服务。

运行方式：
- 模拟版：python src/main.py
- 指定配置：set DOOR_CONFIG=config/door_config.example.json 后运行
- 飞腾派真实 GPIO：把配置 backend 改为 sysfs，或设置 DOOR_BACKEND=sysfs
"""

import os

from ai_gateway import AIGateway
from config import load_config
from door_controller import DoorController
from gpio_mock import MockGPIO
from gpio_sysfs import SysfsGPIO
from logger_setup import setup_logging
from safety import SafetySupervisor
from web_server import DoorWebServer


def create_gpio_backend(config, logger):
    """[BOARD-CONFIG] 根据 backend 选择模拟或真实 GPIO。"""
    backend = str(config.get("backend", "mock")).strip().lower()
    if backend == "mock":
        return MockGPIO(config, logger)
    if backend == "sysfs":
        return SysfsGPIO(config, logger)
    raise ValueError("unsupported backend: %s" % backend)


def main():
    config = load_config()
    logger, memory_handler = setup_logging(config)
    logger.info("loaded config: %s", config.get("_config_path"))
    logger.info("backend=%s", config.get("backend"))

    gpio = create_gpio_backend(config, logger)
    safety = SafetySupervisor(config, gpio, logger)
    controller = DoorController(config, gpio, safety, logger)
    ai_gateway = AIGateway(config, logger)
    web = DoorWebServer(config, controller, ai_gateway, logger, memory_handler)

    controller.start()
    try:
        web.serve_forever()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, shutting down")
    finally:
        web.shutdown()
        controller.shutdown()


if __name__ == "__main__":
    main()
