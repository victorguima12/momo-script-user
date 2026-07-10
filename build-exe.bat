@echo off
chcp 65001 >nul 2>&1
title Build MomoScript.exe
color 0A

REM ============================================================
REM  Build a standalone MomoScript.exe (PyInstaller onedir).
REM
REM  Run this INSIDE the folder you want to build (the admin repo
REM  builds the admin edition; momo-script-user builds the user
REM  edition — deploy-to-user.bat copies this file there).
REM
REM  Output: dist\MomoScript\MomoScript.exe  (+ _internal\)
REM  The USER_EDITION marker (if present) is bundled so the exe
REM  keeps user-mode behavior.
REM ============================================================

cd /d "%~dp0"

python -m PyInstaller --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo PyInstaller not found — installing...
    python -m pip install pyinstaller
)

set "EXTRA="
if exist "USER_EDITION" set "EXTRA=--add-data USER_EDITION;."

python -m PyInstaller --noconfirm --windowed --name MomoScript ^
    --add-data "panel.pt;." %EXTRA% ^
    --collect-all ultralytics ^
    main.py

if %ERRORLEVEL% neq 0 (
    color 0C
    echo.
    echo BUILD FAILED — see output above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Build complete: dist\MomoScript\MomoScript.exe
echo ============================================================
echo.
pause
exit /b 0
