@echo off
echo Testing Python environment...
echo.

echo 1. Creating test Python script...
echo import sys > test.py
echo import os >> test.py
echo print("Python Version:", sys.version) >> test.py
echo print("Executable:", sys.executable) >> test.py
echo print("Working Directory:", os.getcwd()) >> test.py
echo print("Environment Variables:") >> test.py
echo print("  PATH:", os.environ.get('PATH', 'Not set')) >> test.py
echo print("  PYTHONPATH:", os.environ.get('PYTHONPATH', 'Not set')) >> test.py

echo 2. Running Python script...
python test.py > output.txt 2>&1

echo 3. Script output:
type output.txt
echo.

echo 4. Cleaning up...
del test.py

echo.
echo 5. Test complete. Output saved to output.txt
pause
