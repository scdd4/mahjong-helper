@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist "%~dp0dist\雀魂策略助手\雀魂策略助手.exe" (
  start "" "%~dp0dist\雀魂策略助手\雀魂策略助手.exe"
  exit /b 0
)
echo 未找到桌面应用，请保留 dist\雀魂策略助手 整个文件夹。
echo 开发者可按 README.md 中的步骤重新打包。
pause
