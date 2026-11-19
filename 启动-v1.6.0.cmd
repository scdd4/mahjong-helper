@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist "%~dp0dist-v1.6.0\雀魂策略助手\雀魂策略助手.exe" (
  start "" "%~dp0dist-v1.6.0\雀魂策略助手\雀魂策略助手.exe"
  exit /b 0
)
echo 未找到 v1.6.0 桌面应用，请保留 dist-v1.6.0\雀魂策略助手 整个文件夹。
pause
