@echo off
setlocal
cd /d "%~dp0"
set "_PY=python"

if exist venv\Scripts\python.exe (
  venv\Scripts\python.exe -c "import sys" >nul 2>nul
  if not errorlevel 1 set "_PY=venv\Scripts\python.exe"
)

if "%_PY%"=="python" (
  if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe -c "import sys" >nul 2>nul
    if not errorlevel 1 set "_PY=.venv\Scripts\python.exe"
  )
)

%_PY% garic.py %*
