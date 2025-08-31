@echo off
echo Running Python test...
"C:\Users\ayush\ai-coding-assistant\.venv\Scripts\python.exe" test_env.py > test_output.txt 2>&1
type test_output.txt
pause
