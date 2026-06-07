@echo off
chcp 65001 >nul
echo =============================================
echo  SER Demo - Speech Emotion Recognition
echo =============================================
echo.

:: 嘗試找到 venv Python
set VENV_PY=%~dp0venv\Scripts\python.exe

if exist "%VENV_PY%" (
    echo [OK] Using venv Python
    "%VENV_PY%" demo\app.py
) else (
    echo [INFO] venv not found, using system Python
    python demo\app.py
)

echo.
echo [Demo closed]
pause
