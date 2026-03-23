@echo off
setlocal
cd /d "%~dp0"
if exist venv\Scripts\python.exe (
  venv\Scripts\python.exe run_nautilus_browser.py %*
) else (
  python run_nautilus_browser.py %*
)
