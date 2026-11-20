@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist "%~dp0dist-v1.9.0\MahjongStrategyAssistant\MahjongStrategyAssistant.exe" (
  start "" "%~dp0dist-v1.9.0\MahjongStrategyAssistant\MahjongStrategyAssistant.exe"
  exit /b 0
)
echo 未找到桌面应用，请保留 dist-v1.9.0\MahjongStrategyAssistant 整个文件夹。
echo 开发者可按 README.md 中的步骤重新打包。
pause
