# Windows 模拟模式启动脚本：在本机先跑通 Web/状态机/安全逻辑。
# 用法：在项目根目录执行 scripts\run_mock.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$env:DOOR_BACKEND = "mock"
if (-not $env:DOOR_CONFIG) {
  $env:DOOR_CONFIG = "config/door_config.example.json"
}

python src/main.py
