"""
配置读取模块。

作用：
- 从 JSON 读取门控配置。
- 支持环境变量覆盖关键项，避免把 AI key 写进代码或仓库。
- 提供项目根路径解析，方便脚本从不同目录启动。

[BOARD-CONFIG] 板子引脚、端口、超时都必须只在配置文件中改。
"""

import json
import os


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "config", "door_config.example.json")


def load_config(path=None):
    """读取配置文件；path 为空时用 DOOR_CONFIG 环境变量或默认 example 配置。"""
    config_path = path or os.environ.get("DOOR_CONFIG") or DEFAULT_CONFIG
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)
    config_path = os.path.abspath(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    _apply_env_overrides(config)
    config["_config_path"] = config_path
    config["_project_root"] = PROJECT_ROOT
    return config


def _apply_env_overrides(config):
    """只覆盖部署时常改项；不把密钥写入配置文件。"""
    if os.environ.get("DOOR_BACKEND"):
        config["backend"] = os.environ["DOOR_BACKEND"].strip().lower()

    web = config.setdefault("web", {})
    if os.environ.get("DOOR_WEB_HOST"):
        web["host"] = os.environ["DOOR_WEB_HOST"]
    if os.environ.get("DOOR_WEB_PORT"):
        web["port"] = int(os.environ["DOOR_WEB_PORT"])
    if os.environ.get("DOOR_WEB_TOKEN"):
        # [SAFETY-CRITICAL] 控制 token 只从环境变量/配置注入，不要写进前端代码。
        web["control_token"] = os.environ["DOOR_WEB_TOKEN"]

    ai = config.setdefault("ai", {})
    # [AI-ENTRYPOINT] 真实模型 key 只允许从环境变量读取。
    ai["kimi_api_key"] = os.environ.get("KIMI_API_KEY", "")
    ai["glm_api_key"] = os.environ.get("GLM_API_KEY", "")
    ai["openai_compatible_base_url"] = os.environ.get("OPENAI_COMPATIBLE_BASE_URL", "")
    if os.environ.get("DOOR_AI_ENABLED"):
        ai["enabled"] = os.environ["DOOR_AI_ENABLED"].strip().lower() in ("1", "true", "yes", "on")
    if os.environ.get("DOOR_AI_PROVIDER"):
        ai["provider"] = os.environ["DOOR_AI_PROVIDER"].strip().lower()


def resolve_project_path(config, maybe_relative):
    """把配置里的相对路径解析成项目内绝对路径。"""
    if os.path.isabs(maybe_relative):
        return maybe_relative
    return os.path.abspath(os.path.join(config["_project_root"], maybe_relative))
