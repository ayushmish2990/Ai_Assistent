@echo off
echo Resetting virtual environment...

REM Remove existing virtual environment
if exist .venv (
    echo Removing existing virtual environment...
    rmdir /s /q .venv
)

REM Create new virtual environment
echo Creating new virtual environment...
py -3.10 -m venv .venv

REM Install requirements
echo Installing requirements...
call .venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo Failed to activate virtual environment
    exit /b 1
)

pip install --upgrade pip
if %ERRORLEVEL% NEQ 0 (
    echo Failed to upgrade pip
    exit /b 1
)

pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install requirements
    exit /b 1
)

echo.
echo Virtual environment has been reset successfully.
echo Activate it with: .venv\Scripts\activate.bat
pause
