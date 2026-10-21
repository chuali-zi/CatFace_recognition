"""
AI 指令入口。

[AI-ENTRYPOINT] 默认使用本地规则解析；真实大模型接口预留但默认关闭。
[SAFETY-CRITICAL] 无论规则还是大模型，最终都必须转换成 open/close/stop/reset，
再交给 DoorController；AI 不能直接操作 GPIO。
"""

import json
import urllib.request


class AIGateway(object):
    """把自然语言指令安全地转换成 DoorController 命令。"""

    ALLOWED = ("open", "close", "stop", "reset")

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.ai_cfg = config.get("ai", {})

    def execute(self, text, controller):
        """解析文本并调用本地安全 API；返回执行结果。"""
        parsed = self.parse(text)
        command = parsed.get("command")
        if command not in self.ALLOWED:
            return {"ok": False, "error": "unsupported_command", "parsed": parsed}

        # [SAFETY-CRITICAL] 只允许白名单命令进入控制器。
        if command == "open":
            status = controller.open()
        elif command == "close":
            status = controller.close()
        elif command == "stop":
            status = controller.stop()
        else:
            status = controller.reset()
        return {"ok": True, "command": command, "parsed": parsed, "status": status}

    def parse(self, text):
        """优先走真实模型（若启用），否则使用本地规则。"""
        text = (text or "").strip()
        if self.ai_cfg.get("enabled") and self.ai_cfg.get("provider") != "rules":
            model_command = self._call_model(text)
            safe_command = self._normalize_command(model_command)
            if safe_command:
                return {"command": safe_command, "source": "model", "raw": text}
            self.logger.warning("[AI-ENTRYPOINT] model result unavailable/unsafe, fallback to rules")

        return {"command": self._rule_parse(text), "source": "rules", "raw": text}

    def _rule_parse(self, text):
        """简单规则：先满足原型联调，后续再替换为更严谨 NLU。"""
        lowered = text.lower()
        if ("急停" in text) or ("emergency" in lowered):
            return "stop"
        if ("开门" in text) or ("open" in lowered):
            return "open"
        if ("关门" in text) or ("close" in lowered):
            return "close"
        if ("停止" in text) or ("stop" in lowered):
            return "stop"
        if ("复位" in text) or ("reset" in lowered):
            return "reset"
        return None

    def _normalize_command(self, value):
        """[SAFETY-CRITICAL] 模型输出必须落到白名单，其他一律拒绝。"""
        if not value:
            return None
        lowered = str(value).strip().lower()
        if lowered in self.ai_cfg.get("allowed_commands", self.ALLOWED):
            return lowered
        return None

    def _call_model(self, text):
        """[AI-ENTRYPOINT] 预留 OpenAI-compatible 调用；默认返回 None 表示未启用真实模型。

        需要接入时：
        1. 在配置里设 ai.enabled=true、provider=openai_compatible。
        2. 用环境变量提供 OPENAI_COMPATIBLE_BASE_URL 和对应 API KEY。
        3. 让模型只输出 JSON：{"command":"open|close|stop|reset"}。
        """
        provider = self.ai_cfg.get("provider")
        base_url = self.ai_cfg.get("openai_compatible_base_url")
        api_key = self.ai_cfg.get("kimi_api_key") or self.ai_cfg.get("glm_api_key")
        if provider != "openai_compatible" or not base_url or not api_key:
            return None

        # 这里故意保持保守：没有完整 prompt/重试/审计前，不默认发起网络请求。
        self.logger.info("[AI-ENTRYPOINT] real model call is reserved but not implemented")
        return None
