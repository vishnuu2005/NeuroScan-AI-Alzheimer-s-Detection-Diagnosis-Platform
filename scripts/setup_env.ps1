# NeuroScan AI - Windows 11 Environment Setup Script
# Configures a clean virtual environment using the host Python installation

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " NeuroScan AI: Automated Windows 11 Setup" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Verify Python availability
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Error "Python was not found in PATH. Please install Python 3.10+ (64-bit)."
    exit 1
}

$pyVersion = python --version
Write-Host "[1/4] Found $pyVersion" -ForegroundColor Green

# 2. Create virtual environment if absent
$venvPath = Join-Path $PSScriptRoot "..\.venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "[2/4] Creating virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv $venvPath --system-site-packages
} else {
    Write-Host "[2/4] Virtual environment exists at $venvPath" -ForegroundColor Green
}

# 3. Upgrade pip and install requirements
$pipPath = Join-Path $venvPath "Scripts\pip.exe"
Write-Host "[3/4] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
& $pipPath install -r (Join-Path $PSScriptRoot "..\requirements.txt")

# 4. Verify directory structures
Write-Host "[4/4] Verifying required runtime directories..." -ForegroundColor Yellow
$dirs = @(
    (Join-Path $PSScriptRoot "..\models"),
    (Join-Path $PSScriptRoot "..\chatbot\static\uploads"),
    (Join-Path $PSScriptRoot "..\chatbot\static\reports"),
    (Join-Path $PSScriptRoot "..\chatbot\instance"),
    (Join-Path $PSScriptRoot "..\.matplotlib_cache")
)

foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  + Created $dir" -ForegroundColor DarkGray
    }
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Setup complete! Activate via: .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan
