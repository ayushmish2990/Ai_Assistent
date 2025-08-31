@echo off
echo Testing Python environment...
echo.

echo 1. Checking Python version:
py --version
echo.

echo 2. Checking Python executable:
where python
echo.

echo 3. Running a simple Python command:
echo print("Hello, World!") > test.py
python test.py
if exist test.py del test.py
echo.

echo 4. Checking Python modules:
python -c "import sys; print('Python path:', sys.path)"
echo.

echo 5. Checking pip:
pip --version
echo.

pause
