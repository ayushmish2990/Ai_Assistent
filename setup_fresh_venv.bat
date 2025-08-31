@echo off
setlocal enabledelayedexpansion

echo Setting up a fresh Python virtual environment...
echo.

REM Remove existing virtual environment
if exist .venv (
    echo Removing existing virtual environment...
    rmdir /s /q .venv
)

echo.
echo Step 1: Checking Python installation...
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python is not in PATH. Please install Python 3.10 or add it to your PATH.
    pause
    exit /b 1
)

python --version
if %ERRORLEVEL% NEQ 0 (
    echo Failed to get Python version. Please check your Python installation.
    pause
    exit /b 1
)

echo.
echo Step 2: Creating new virtual environment...
python -m venv .venv
if %ERRORLEVEL% NEQ 0 (
    echo Failed to create virtual environment.
    pause
    exit /b 1
)

echo.
echo Step 3: Activating virtual environment...
call .venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)

echo.
echo Step 4: Upgrading pip...
python -m pip install --upgrade pip
if %ERRORLEVEL% NEQ 0 (
    echo Failed to upgrade pip.
    pause
    exit /b 1
)

echo.
echo Step 5: Installing requirements...
if exist requirements.txt (
    pip install -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to install requirements.
        pause
        exit /b 1
    )
) else (
    echo requirements.txt not found. Skipping requirements installation.
)

echo.
echo Virtual environment setup complete!
echo.
echo To activate the virtual environment, run:
echo    .venv\Scripts\activate.bat
echo.
pause
