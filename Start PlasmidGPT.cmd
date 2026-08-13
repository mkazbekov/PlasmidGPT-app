@echo off
setlocal
cd /d "%~dp0"
title PlasmidGPT Studio

set "PLASMIDGPT_PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%PLASMIDGPT_PYTHON%" (
  echo.
  echo  First-time setup: creating PlasmidGPT's local Python environment...
  echo  This normally takes a few minutes and happens only once.
  echo.

  where py >nul 2>&1
  if not errorlevel 1 (
    py -3.10 -m venv "%~dp0.venv" 2>nul || py -3 -m venv "%~dp0.venv"
  ) else (
    where python >nul 2>&1
    if errorlevel 1 (
      echo.
      echo  Python was not found on this computer.
      echo  Install Python 3.10+ from https://www.python.org/downloads/
      echo  and make sure to check "Add python.exe to PATH" during setup,
      echo  then double-click this file again.
      echo.
      pause
      exit /b 1
    )
    python -m venv "%~dp0.venv"
  )

  if not exist "%PLASMIDGPT_PYTHON%" (
    echo.
    echo  Could not create the Python environment. See any messages above.
    echo.
    pause
    exit /b 1
  )
)

"%PLASMIDGPT_PYTHON%" -c "import streamlit" >nul 2>&1
if errorlevel 1 (
  echo.
  echo  First-time setup: installing PlasmidGPT's dependencies...
  echo  This normally takes a few minutes and happens only once.
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
  "%PLASMIDGPT_PYTHON%" -m pip install --upgrade pip >nul
  "%PLASMIDGPT_PYTHON%" -m pip install -r "%~dp0requirements-app.txt"
  if errorlevel 1 (
    echo.
    echo  Setup could not finish. Check your internet connection and try again.
    echo.
    pause
    exit /b 1
  )
)

where nvidia-smi >nul 2>&1
if not errorlevel 1 (
  "%PLASMIDGPT_PYTHON%" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
  if errorlevel 1 (
    echo.
    echo  NVIDIA GPU detected - installing the CUDA-enabled PyTorch build...
    echo  This only needs to happen once.
    echo.
    "%PLASMIDGPT_PYTHON%" -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121
  )
)

if not exist "%~dp0pretrained_model\pretrained_model.pt" (
  echo.
  echo  The pretrained model file is missing:
  echo    pretrained_model\pretrained_model.pt
  echo.
  echo  If you downloaded this project as a ZIP file, Git LFS assets are not
  echo  included. Clone it with Git instead so large files download correctly:
  echo.
  echo    git lfs install
  echo    git clone https://github.com/mkazbekov/PlasmidGPT-app.git
  echo.
  pause
  exit /b 1
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
