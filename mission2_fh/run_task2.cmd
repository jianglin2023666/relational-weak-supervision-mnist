@echo off
cd /d "%~dp0.."
set PYTHONUTF8=1
".venv\Scripts\python.exe" -u "%~dp0run_task2.py" %*
if errorlevel 1 exit /b 1
pause
