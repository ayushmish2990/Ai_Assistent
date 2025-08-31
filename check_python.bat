@echo off
setlocal enabledelayedexpansion

set PYTHON_EXE="C:\Program Files\Python310\python.exe"

echo Using Python from: %PYTHON_EXE%

%PYTHON_EXE% --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python not found at the specified path.
    pause
    exit /b 1
)

echo.
echo Creating virtual environment...
%PYTHON_EXE% -m venv .venv
if %ERRORLEVEL% NEQ 0 (
    echo Failed to create virtual environment.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)

echo.
echo Installing required packages from requirements.txt...
python -m pip install --upgrade pip
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install required packages.
    pause
    exit /b 1
)

echo.
echo Setup complete! Virtual environment is ready to use.
echo.
echo To activate the virtual environment later, run:
echo   call .venv\Scripts\activate.bat
echo.
pause
