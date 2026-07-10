@echo off
chcp 65001 >nul 2>&1
title Momo Script - Launcher
color 0A

echo ============================================================
echo              MOMO SCRIPT - Auto Launcher
echo ============================================================
echo.

REM --- Navigate to script directory ---
cd /d "%~dp0"

REM ============================================================
REM  STEP 1: Check if Python is installed
REM ============================================================
echo [1/4] Checking Python installation...
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    color 0C
    echo.
    echo ============================================================
    echo  ERROR: Python is NOT installed or not in PATH.
    echo ============================================================
    echo.
    echo  WHAT THIS MEANS:
    echo    This app needs Python to run, but your computer
    echo    either doesn't have it or can't find it.
    echo.
    echo  HOW TO FIX:
    echo    1. Go to https://www.python.org/downloads/
    echo    2. Download Python 3.11 or newer
    echo    3. During install, CHECK the box that says
    echo       "Add Python to PATH" (very important!)
    echo    4. Restart your computer after installing
    echo    5. Double-click this .bat file again
    echo.
    echo  COPY THIS FOR AI HELP:
    echo    "Momo Script failed: Python not found in PATH on Windows 11.
    echo     I need help installing Python 3.11+ and adding it to PATH."
    echo ============================================================
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo    Found Python %PYVER% - OK
echo.

REM ============================================================
REM  STEP 2: Check if pip is available
REM ============================================================
echo [2/4] Checking pip...
python -m pip --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    color 0C
    echo.
    echo ============================================================
    echo  ERROR: pip is NOT available.
    echo ============================================================
    echo.
    echo  WHAT THIS MEANS:
    echo    pip is the tool that installs Python packages.
    echo    It should come with Python but something went wrong.
    echo.
    echo  HOW TO FIX:
    echo    Run this command in a terminal:
    echo      python -m ensurepip --upgrade
    echo    Then double-click this .bat file again.
    echo.
    echo  COPY THIS FOR AI HELP:
    echo    "Momo Script failed: pip not found. I have Python %PYVER%
    echo     on Windows 11 but pip is missing."
    echo ============================================================
    pause
    exit /b 1
)
echo    pip is available - OK
echo.

REM ============================================================
REM  STEP 3: Install/update dependencies
REM ============================================================
echo [3/4] Installing dependencies (this may take a few minutes on first run)...
echo    Installing from requirements.txt...
echo.

python -m pip install -r requirements.txt --quiet 2>pip_errors.tmp
python -m pip install selenium webdriver-manager --quiet 2>>pip_errors.tmp
if %ERRORLEVEL% neq 0 (
    color 0C
    echo.
    echo ============================================================
    echo  ERROR: Failed to install some dependencies.
    echo ============================================================
    echo.
    echo  WHAT THIS MEANS:
    echo    Some Python packages couldn't be installed.
    echo    This is usually a network or permissions issue.
    echo.
    echo  ERROR DETAILS:
    type pip_errors.tmp
    echo.
    echo  HOW TO FIX:
    echo    - Make sure you have internet access
    echo    - Try running this .bat as Administrator
    echo      (right-click ^> Run as administrator)
    echo    - If you see "Microsoft Visual C++" errors, install
    echo      Visual Studio Build Tools from:
    echo      https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo.
    echo  COPY THIS FOR AI HELP:
    echo    "Momo Script failed: pip install failed for requirements.txt.
    echo     Python %PYVER%, Windows 11. Error output above."
    echo ============================================================
    del pip_errors.tmp 2>nul
    pause
    exit /b 1
)
del pip_errors.tmp 2>nul
echo    All dependencies installed - OK
echo.

REM ============================================================
REM  STEP 4: Check that the YOLO model file exists
REM ============================================================
echo [4/4] Checking required files...
if not exist "panel.pt" (
    color 0E
    echo.
    echo  WARNING: panel.pt (YOLO model) not found.
    echo  Panel auto-detection will NOT work.
    echo  The app will still open, but you'll need to
    echo  create boxes manually.
    echo.
)

if not exist "main.py" (
    color 0C
    echo.
    echo ============================================================
    echo  ERROR: main.py not found in this folder.
    echo ============================================================
    echo.
    echo  WHAT THIS MEANS:
    echo    The main application file is missing.
    echo    The .bat file must be in the same folder as main.py.
    echo.
    echo  CURRENT FOLDER: %cd%
    echo.
    echo  COPY THIS FOR AI HELP:
    echo    "Momo Script failed: main.py not found in %cd%.
    echo     The START.bat can't find the application files."
    echo ============================================================
    pause
    exit /b 1
)
echo    All files present - OK
echo.

REM ============================================================
REM  LAUNCH THE APP
REM ============================================================
echo ============================================================
echo  All checks passed! Launching Momo Script...
echo  (this window will stay open to show errors)
echo ============================================================
echo.

python main.py 2>app_errors.tmp
set APP_EXIT=%ERRORLEVEL%

if %APP_EXIT% neq 0 (
    color 0C
    echo.
    echo ============================================================
    echo  ERROR: Momo Script crashed (exit code %APP_EXIT%)
    echo ============================================================
    echo.
    echo  ERROR DETAILS:
    type app_errors.tmp
    echo.
    echo  COMMON FIXES:
    echo    - If "No module named X": a package is missing.
    echo      Run: python -m pip install X
    echo    - If "DLL load failed": reinstall the failing package.
    echo      Run: python -m pip install --force-reinstall X
    echo    - If "CUDA" or "GPU" error: this is a graphics driver
    echo      issue. Update your GPU drivers.
    echo.
    echo  COPY THIS FOR AI HELP:
    echo    "Momo Script crashed with exit code %APP_EXIT% on Windows 11,
    echo     Python %PYVER%. Error details shown above."
    echo ============================================================
)
del app_errors.tmp 2>nul

echo.
echo Press any key to close...
pause >nul
