# NeuroScan AI · Alzheimer's Detection & Clinical Diagnosis Platform

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Flask-3.0%2B-green.svg)](https://flask.palletsprojects.com/)
[![Deep Learning](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)

**NeuroScan AI** is an intelligent, full-stack medical decision-support platform designed to analyze brain MRI scans, classify Alzheimer's disease progression stages, generate comprehensive clinical PDF reports, and provide an interactive conversational AI assistant for clinical education and caregiver guidance.

---

## Key Features

- **Deep Learning MRI Classification**: Custom single-channel ResNet-18 architecture fine-tuned to classify scans into 4 stages:
  - *NonDemented* (No cognitive impairment detected)
  - *VeryMildDemented* (Early mild cognitive impairment)
  - *MildDemented* (Mild dementia)
  - *ModerateDemented* (Moderate dementia)
- **Uncertainty Quantification**: Calculates prediction confidence and normalized Shannon entropy to flag low-confidence or high-entropy scans for mandatory human neurologist review.
- **AI-Powered Clinical Chatbot**: Powered by Groq's high-speed Llama-3.3-70B model with context-aware prompt engineering and an intelligent offline fallback engine.
- **Automated Clinical Reports**: Generates downloadable PDF reports (using ReportLab) featuring clinical staging details, probability distribution breakdowns, and patient metadata.
- **Security & Performance**: Hardened session cookies (`HttpOnly`, `SameSite=Lax`), UUID upload sanitization, 16MB `MAX_CONTENT_LENGTH` enforcement, and database query indexing.
- **Modern Responsive Dashboard**: Real-time analytics, stage filtering, and mobile-responsive dark theme interface.

---

## Architecture Overview

```
NeuroScan AI
├── chatbot/                 # Web application & API layer
│   ├── app.py               # Flask application & route controllers
│   ├── generate_reports.py  # ReportLab PDF clinical generator
│   ├── inference.py         # PyTorch inference, validation & entropy
│   ├── validators.py        # Input sanitization & file constraints
│   ├── templates/           # Jinja2 dashboard, login & scan templates
│   └── static/              # CSS stylesheets, client scripts & uploads
├── models/                  # Neural network architecture definitions
│   └── resnet_model.py      # 1-channel modified ResNet-18
├── utils/                   # Data preprocessing & path configuration
│   ├── config.py            # Centralized dynamic path manager
│   ├── data_preprocessing.py# PyTorch dataset loaders & transforms
│   └── augment_and_balance.py # Minority class balancing
├── tests/                   # Automated unit & integration test suite
└── .github/workflows/       # GitHub Actions CI matrix & linting
```

---

## Quickstart Guide

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 2. Installation

Clone the repository and create an isolated virtual environment:

```bash
git clone https://github.com/vishnuu2005/NeuroScan-AI-Alzheimer-s-Detection-Diagnosis-Platform.git
cd NeuroScan-AI-Alzheimer-s-Detection-Diagnosis-Platform

python -m venv .venv
# On Windows:
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy the sample environment template:

```bash
cp .env.example .env
```

| Variable | Default | Description |
| :--- | :--- | :--- |
| `FLASK_SECRET_KEY` | *(Random)* | Secret key for cryptographic session signing |
| `FLASK_ENV` | `development` | Environment mode (`development` or `production`) |
| `GROQ_API_KEY` | `""` | API key from [Groq Console](https://console.groq.com/) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | LLM model identifier for clinical reasoning |
| `USE_GROQ` | `1` | Set to `0` to run exclusively in offline rule-based mode |
| `PREDICTION_THRESHOLD`| `0.30` | Minimum confidence threshold before flagging as *Uncertain* |
| `MAX_CONTENT_LENGTH` | `16777216` | Maximum file upload size in bytes (16 MB) |

### 4. Running the Application

Run the application module from the project root:

```bash
python -m chatbot.app
```

Navigate to `http://127.0.0.1:5000` in your web browser.

---

## Running Automated Tests

Execute the unit and integration test suite:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## Medical Disclaimer

**IMPORTANT NOTICE**: NeuroScan AI is developed for research, education, and adjunctive clinical decision-support purposes only. It does NOT constitute a medical device or formal clinical diagnosis. Classification results must always be reviewed and confirmed by a qualified, board-certified neurologist or radiologist.
