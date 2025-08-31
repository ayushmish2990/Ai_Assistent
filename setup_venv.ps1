# Create virtual environment
Write-Host "Creating Python 3.10 virtual environment..."
python3.10 -m venv .venv

# Activate the virtual environment
Write-Host "Activating virtual environment..."
.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

# Install required packages
Write-Host "Installing required packages..."
pip install torch==2.0.1 transformers==4.30.2 pytest

# Verify installation
Write-Host "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); import transformers; print(f'Transformers version: {transformers.__version__}')"

Write-Host "`nSetup complete! Virtual environment is ready to use.`n"
Write-Host "To activate the virtual environment later, run:"
Write-Host ".\.venv\Scripts\Activate.ps1"
