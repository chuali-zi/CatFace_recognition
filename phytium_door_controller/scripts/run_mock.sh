#!/usr/bin/env bash
# Linux/macOS 模拟模式启动脚本：用于无飞腾派硬件时先验证软件流程。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export DOOR_BACKEND=mock
export DOOR_CONFIG="${DOOR_CONFIG:-config/door_config.example.json}"
python3 src/main.py
