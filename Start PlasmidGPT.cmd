@echo off
setlocal
cd /d "%~dp0"
title PlasmidGPT Studio

set "PLASMIDGPT_PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%PLASMIDGPT_PYTHON%" (
  echo.
  echo  PlasmidGPT's Python environment was not found.
  echo  Expected: %PLASMIDGPT_PYTHON%
  echo.
  echo  Reinstall the local environment, then double-click this file again.
  echo.
  pause
  exit /b 1
)

"%PLASMIDGPT_PYTHON%" -c "import streamlit" >nul 2>&1
if errorlevel 1 (
  echo.
  echo  First-time setup: installing the local interface...
  echo  This normally takes one or two minutes and happens only once.
  echo.
  "%PLASMIDGPT_PYTHON%" -m pip --version >nul 2>&1
  if errorlevel 1 (
    "%PLASMIDGPT_PYTHON%" -m ensurepip --upgrade --default-pip
    if errorlevel 1 (
      echo.
      echo  Python's package installer could not be prepared.
      echo.
      pause
      exit /b 1
    )
  )
  "%PLASMIDGPT_PYTHON%" -m pip install -r "%~dp0requirements-app.txt"
  if errorlevel 1 (
    echo.
    echo  Setup could not finish. Check your internet connection and try again.
    echo.
    pause
    exit /b 1
  )
)

echo.
echo  Starting PlasmidGPT Studio...
echo  Your browser will open automatically.
echo  Keep this window open while using PlasmidGPT.
echo.

"%PLASMIDGPT_PYTHON%" -m streamlit run "%~dp0app.py"

if errorlevel 1 (
  echo.
  echo  PlasmidGPT stopped unexpectedly. Review the message above.
  echo.
  pause
)

endlocal
