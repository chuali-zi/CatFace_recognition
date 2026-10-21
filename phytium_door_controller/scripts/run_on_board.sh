#!/usr/bin/env bash
# 飞腾派真实 GPIO 启动脚本：切换到 sysfs 后端。
# [GPIO-SYSFS] /sys/class/gpio 通常需要 root 权限；也可用 udev/用户组方案替代 sudo。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export DOOR_BACKEND=sysfs
export DOOR_CONFIG="${DOOR_CONFIG:-config/door_config.example.json}"

# [HARDWARE-TODO] 若正式产品不想用 sudo，请配置 GPIO 权限/udev 规则后去掉 sudo。
sudo -E python3 src/main.py
