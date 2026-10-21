"""
Web 服务模块：提供本地 Web API 和静态控制页面。

只用 Python 标准库 http.server，方便在飞腾派 Ubuntu 20.04/Python 3.8 上运行。
API：
- GET  /api/status
- POST /api/open
- POST /api/close
- POST /api/stop
- POST /api/reset
- POST /api/ai-command
"""

import hmac
import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from config import resolve_project_path


class DoorWebServer(object):
    """封装 HTTP 服务，main.py 只负责启动/停止。"""

    def __init__(self, config, controller, ai_gateway, logger, memory_handler):
        self.config = config
        self.controller = controller
        self.ai_gateway = ai_gateway
        self.logger = logger
        self.memory_handler = memory_handler
        self.httpd = None

    def serve_forever(self):
        web_cfg = self.config.get("web", {})
        host = web_cfg.get("host", "127.0.0.1")
        port = int(web_cfg.get("port", 8080))
        static_dir = resolve_project_path(self.config, web_cfg.get("static_dir", "static"))

        handler_class = self._make_handler(static_dir)
        self.httpd = ThreadingHTTPServer((host, port), handler_class)
        if not str(web_cfg.get("control_token", "")):
            self.logger.warning("web.control_token 为空：POST /api/* 不鉴权，仅适合本机或实验环境")
        self.logger.info("Web server listening on http://%s:%s", host, port)
        self.httpd.serve_forever()

    def shutdown(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()

    def _make_handler(self, static_dir):
        controller = self.controller
        ai_gateway = self.ai_gateway
        logger = self.logger
        memory_handler = self.memory_handler
        web_cfg = self.config.get("web", {})
        log_limit = int(web_cfg.get("log_tail_limit", 50))
        control_token = str(web_cfg.get("control_token", ""))

        class Handler(BaseHTTPRequestHandler):
            def _json(self, code, payload):
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _read_json(self):
                length = int(self.headers.get("Content-Length", "0") or "0")
                if length <= 0:
                    return {}
                raw = self.rfile.read(length).decode("utf-8")
                return json.loads(raw or "{}")

            def do_GET(self):
                path = urlparse(self.path).path
                if path == "/api/status":
                    payload = controller.status()
                    payload["logs"] = memory_handler.tail(log_limit)
                    self._json(200, payload)
                    return
                self._serve_static(path)

            def do_POST(self):
                path = urlparse(self.path).path
                if not self._control_token_ok(path):
                    self._json(403, {"ok": False, "error": "forbidden_bad_token"})
                    return
                try:
                    if path == "/api/open":
                        self._json(200, controller.open())
                    elif path == "/api/close":
                        self._json(200, controller.close())
                    elif path == "/api/stop":
                        self._json(200, controller.stop())
                    elif path == "/api/reset":
                        self._json(200, controller.reset())
                    elif path == "/api/ai-command":
                        body = self._read_json()
                        result = ai_gateway.execute(body.get("text", ""), controller)
                        self._json(200 if result.get("ok") else 400, result)
                    else:
                        self._json(404, {"ok": False, "error": "not_found"})
                except Exception as exc:  # Web 层兜底，防止一次请求拖垮服务
                    logger.exception("API error path=%s", path)
                    self._json(500, {"ok": False, "error": str(exc)})

            def _control_token_ok(self, path):
                """机械门控制接口鉴权：token 非空时所有 POST /api/* 都必须匹配。"""
                if not path.startswith("/api/"):
                    return True
                if not control_token:
                    return True
                given = self.headers.get("X-Door-Token", "")
                return hmac.compare_digest(given, control_token)

            def _serve_static(self, path):
                if path in ("/", ""):
                    path = "/index.html"
                static_root = os.path.abspath(static_dir)
                safe_path = os.path.normpath(path).lstrip("\\/")
                file_path = os.path.abspath(os.path.join(static_root, safe_path))
                if os.path.commonpath([static_root, file_path]) != static_root or not os.path.isfile(file_path):
                    self._json(404, {"ok": False, "error": "static_not_found"})
                    return

                with open(file_path, "rb") as f:
                    data = f.read()
                ctype = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, fmt, *args):
                logger.info("HTTP %s", fmt % args)

        return Handler
