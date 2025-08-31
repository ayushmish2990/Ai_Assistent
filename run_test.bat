@echo off
setlocal enabledelayedexpansion

set PYTHON_CMD=".venv\Scripts\python.exe"
set SCRIPT="test_imports.py"
set OUTPUT_FILE="test_output.txt"

echo Running %PYTHON_CMD% %SCRIPT%
%PYTHON_CMD% %SCRIPT% > %OUTPUT_FILE% 2>&1

echo.
echo Script execution complete. Output saved to %OUTPUT_FILE%
echo.
type %OUTPUT_FILE%

echo.
echo Press any key to exit...
pause > nul
