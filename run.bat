@echo off
cd /d "%~dp0"
python main.py 2> stderr.log
pause
