# NeuroScan AI - Application Launcher (Windows 11)
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location -Path $ProjectRoot

# Set Matplotlib cache dir to project folder to prevent Windows permission errors
$env:MPLCONFIGDIR = Join-Path $ProjectRoot ".matplotlib_cache"
$env:PYTHONPATH = $ProjectRoot

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "Virtual environment not found. Using system python..." -ForegroundColor Yellow
    $VenvPython = "python"
}

Write-Host "Starting NeuroScan AI on http://127.0.0.1:5000 ..." -ForegroundColor Cyan
& $VenvPython -m chatbot.app
