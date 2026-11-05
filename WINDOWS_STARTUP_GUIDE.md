# NeuroScan AI — Windows 11 Development & Startup Guide

This document provides complete, step-by-step instructions for installing, configuring, running, and verifying **NeuroScan AI** on Windows 11 and Windows 10 systems.

---

## 1. Prerequisites

Before starting, ensure your system has the following installed:

- **Operating System:** Windows 11 64-bit (or Windows 10 build 19041+)
- **Python:** Python 3.10 through 3.14 (Python 3.14 recommended)
  - During Python installation, check **"Add python.exe to PATH"**
- **Git:** Git for Windows (`git --version` >= 2.40)
- **PowerShell:** Windows PowerShell 5.1 or PowerShell 7+
- **Hardware:**
  - Minimum: 8 GB RAM, 4-core CPU, 2 GB free disk space
  - Recommended: 16 GB RAM, dedicated NVIDIA GPU (optional for CUDA acceleration)

---

## 2. Repository Setup

Clone the primary repository into your local workspace:

```powershell
git clone https://github.com/vishnuu2005/NeuroScan-AI-Alzheimer-s-Detection-Diagnosis-Platform.git
cd NeuroScan-AI-Alzheimer-s-Detection-Diagnosis-Platform
```

---

## 3. Python Virtual Environment Configuration

Always develop inside a dedicated virtual environment to prevent dependency collisions:

```powershell
# Create the virtual environment
python -m venv .venv

# Enable script execution policy for the current session (if restricted)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

Verify that the active Python points to `.venv`:
```powershell
Get-Command python | Select-Object -ExpandProperty Source
# Expected output: ...\NeuroScan-AI-Alzheimer-s-Detection-Diagnosis-Platform\.venv\Scripts\python.exe
```

---

## 4. Install Dependencies

Install all pinned packages and PyTorch:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify PyTorch and key clinical packages:
```powershell
python -c "import torch, torchvision, flask, reportlab; print('PyTorch Version:', torch.__version__, '| CUDA Available:', torch.cuda.is_available())"
```

---

## 5. Environment Configuration (.env)

Copy the environment template and configure your secrets:

```powershell
Copy-Item .env.example .env
```

Open `.env` in your editor and configure the following variables:

```ini
# Flask Secret Key
FLASK_SECRET_KEY=neuroscan_production_secure_token_change_me

# Database URI (SQLite defaults to instance/neuroscan.db)
SQLALCHEMY_DATABASE_URI=sqlite:///instance/neuroscan.db

# Groq LLaMA 3.3 Clinical Consultation (Optional)
GROQ_API_KEY=gsk_your_groq_api_key_here
USE_GROQ=true

# Model Weights Configuration
MODEL_PATH=models/resnet2d_alzheimers_best.pth
PREDICTION_THRESHOLD=0.30
IMAGE_SIZE=224

# Matplotlib Local Cache (prevents Windows Temp permission warnings)
MPLCONFIGDIR=.matplotlib_cache
```

---

## 6. Model Checkpoint Verification

Ensure the trained ResNet-18 model checkpoint is located in the `models/` directory:

```powershell
Test-Path models\resnet2d_alzheimers_best.pth
```

If using a custom checkpoint, update `MODEL_PATH` in `.env`. The adaptive loader dynamically inspects whether the checkpoint was trained on 1-channel (grayscale) or 3-channel (RGB) inputs and adjusts the convolutional backbone automatically.

---

## 7. Running the Application

### Development Server:
```powershell
# Run the Flask development server
python -m chatbot.app
```
Or:
```powershell
$env:FLASK_APP = "chatbot.app"
flask run --host=127.0.0.1 --port=5000
```

Open your browser to:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

### Production WSGI Server (Waitress):
For production-grade concurrency on Windows without development server warnings:
```powershell
pip install waitress
waitress-serve --listen=127.0.0.1:5000 chatbot.app:app
```

---

## 8. Running the Automated Test Suite

NeuroScan AI features comprehensive test harnesses covering security, AI inference, reporting, database, and APIs:

```powershell
# Run all test suites
.\.venv\Scripts\pytest.exe

# Run with verbose output and coverage report
.\.venv\Scripts\pytest.exe -v --cov=chatbot

# Run specific functional suites
.\.venv\Scripts\pytest.exe tests/test_security.py
.\.venv\Scripts\pytest.exe tests/test_inference.py
.\.venv\Scripts\pytest.exe tests/test_perf.py
```

---

## 9. Windows 11 Troubleshooting FAQ

### Q1: `Execution of scripts is disabled on this system`
**Fix:** Run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Q2: `Matplotlib is building the font cache; this may take a moment`
**Fix:** NeuroScan AI automatically points `MPLCONFIGDIR` to a workspace `.matplotlib_cache/` folder. Ensure this folder has write permissions.

### Q3: `Port 5000 is already in use`
**Fix:** Identify and stop the occupying process:
```powershell
Get-NetTCPConnection -LocalPort 5000 | Select-Object OwningProcess
Stop-Process -Id <PID> -Force
```

### Q4: Line ending warnings (`LF will be replaced by CRLF`)
**Fix:** Configure Git to normalize line endings:
```powershell
git config core.autocrlf true
```
