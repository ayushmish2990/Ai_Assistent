# Remove existing virtual environment
if (Test-Path -Path .\.venv) {
    Write-Host "Removing existing virtual environment..."
    Remove-Item -Recurse -Force .\.venv
}

# Create new virtual environment
Write-Host "Creating new virtual environment..."
py -3.10 -m venv .venv

# Activate and install requirements
Write-Host "Installing requirements..."
.\venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

Write-Host "Virtual environment has been reset successfully."
Write-Host "Activate it with: .\.venv\Scripts\Activate.ps1"
