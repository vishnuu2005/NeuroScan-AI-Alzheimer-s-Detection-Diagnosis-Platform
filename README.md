# NeuroScan AI — Clinical Alzheimer's MRI Detection & Consultation Platform

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1%2B-black.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Security](https://img.shields.io/badge/OWASP-Hardened-emerald.svg)]()
[![Build](https://img.shields.io/badge/Tests-Passing%20(85%2B)-success.svg)]()

**NeuroScan AI** is an advanced clinical decision-support platform designed to analyze axial brain Magnetic Resonance Imaging (MRI) scans, classify Alzheimer’s disease progression stages, generate comprehensive medical PDF diagnostic reports, and provide interactive clinical consultation via Groq LLaMA 3.3 70B with intelligent offline rule-engine fallbacks.

---

## 🏗 System Architecture

```text
                                  +---------------------------------------+
                                  |         Clinician Web Client          |
                                  |  (HTML5 / Chart.js / FontAwesome 6)   |
                                  +-------------------+-------------------+
                                                      |
                                                      v  HTTP / REST API / CSP
+-------------------------------------------------------------------------------------------------------+
|                                        NeuroScan Flask Backend                                        |
|                                                                                                       |
|  +--------------------+   +-----------------------+   +--------------------+   +-------------------+  |
|  |   Authentication   |   |   MIME / Byte Magic   |   |  Security Headers  |   | Rotating Clinical |  |
|  |  (Scrypt / Timing) |   |  Upload Verification  |   |   (CSP / HSTS)     |   |   Audit Logger    |  |
|  +--------------------+   +-----------------------+   +--------------------+   +-------------------+  |
|                                       |                                                               |
|                                       v                                                               |
|  +-------------------------------------------------------------------------------------------------+  |
|  |                                AI Diagnostic & Inference Engine                                 |  |
|  |  +-------------------------------+   +-----------------------------+   +---------------------+  |  |
|  |  | Adaptive ResNet-18 ConvNet    |   | Probability Distribution    |   | Model JIT Warmup    |  |  |
|  |  | (1-Channel / 3-Channel Auto)  |   | & Clinical Risk Tier Engine |   | & Latency Profiling |  |  |
|  |  +-------------------------------+   +-----------------------------+   +---------------------+  |  |
|  +-------------------------------------------------------------------------------------------------+  |
|                                       |                                                               |
|                 +---------------------+---------------------+                                         |
|                 v                                           v                                         |
|  +-----------------------------+             +-----------------------------+                          |
|  |   Groq Clinical AI Engine   |             | In-Memory ReportLab Engine  |                          |
|  | (LLaMA 3.3 70B + Fallback)  |             |  (Streamed Hospital PDF)    |                          |
|  +-----------------------------+             +-----------------------------+                          |
+-------------------------------------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------------------------------+
|                                    Persistent Storage Layer                                           |
|       SQLite / PostgreSQL  &bull;  Composite Indexes (User, Scan, Session, CreatedAt)               |
+-------------------------------------------------------------------------------------------------------+
```

---

## 🧠 Clinical Classification & GDS Staging

NeuroScan AI categorizes MRI scans into four progressive stages corresponding to the validated **Global Deterioration Scale (GDS)**:

| Model Category | Clinical Terminology | GDS Stage | Clinical Characteristics | Clinical Risk Tier |
|:---|:---|:---|:---|:---|
| `NonDemented` | Non-Demented | GDS 1–2 | Normal cortical thickness and hippocampal volume. No cognitive deficit. | `LOW RISK` |
| `VeryMildDemented` | Very Mild Cognitive Impairment | GDS 3 | Subtle memory lapses, minor spatial disorientation, preserved daily functioning. | `EARLY STAGE` |
| `MildDemented` | Mild Dementia | GDS 4–5 | Noticeable memory deficits, impaired complex task management, temporal confusion. | `MODERATE RISK` |
| `ModerateDemented` | Moderate Dementia | GDS 6–7 | Severe cognitive decline, hippocampal atrophy, language impairment, extensive care needed. | `HIGH RISK` |
| `Uncertain` | Inconclusive / Low Confidence | — | Probability falls below confidence threshold (`0.30`). Neurologist consultation mandated. | `UNKNOWN` |

---

## ⚡ Key Platform Capabilities

1. **Adaptive ResNet-18 Architecture:**
   - Dynamically inspects model checkpoints to accommodate both 1-channel (grayscale) and 3-channel (RGB) weights.
   - Eliminates checkpoint shape mismatches with strict `state_dict` reflection.
   - Startup kernel warmup pass eliminates first-request cold starts.

2. **Dual-Mode Clinical AI Consultation:**
   - Powered by Groq LLaMA 3.3 70B with expert neuro-clinical system prompting.
   - Robust offline fallback driven by rule-based GDS staging and evidence-based diagnostic protocols.

3. **In-Memory Clinical PDF Report Generator:**
   - ReportLab-powered hospital layout with clinical headers, patient reference identifiers, multi-class distribution graphs, and physician sign-off block.
   - Direct memory streaming via `io.BytesIO` eliminates server disk I/O bottlenecks.

4. **Hardened Security Architecture:**
   - OWASP Security Headers: Strict Content Security Policy (CSP), `X-Frame-Options: SAMEORIGIN`, `nosniff`, `Referrer-Policy`.
   - MIME type and magic-byte header validation defending against malicious payload spoofing.
   - Timing-attack-resistant authentication with constant-time password hash verification.
   - Strict directory traversal protection in upload and report file routing.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10 to 3.14
- Git
- Windows 11 / Linux (Ubuntu 22.04+) / macOS

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/vishnuu2005/NeuroScan-AI-Alzheimer-s-Detection-Diagnosis-Platform.git
cd NeuroScan-AI-Alzheimer-s-Detection-Diagnosis-Platform

# Create and activate virtual environment
python -m venv .venv

# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
```bash
cp .env.example .env
# Edit .env to set your FLASK_SECRET_KEY and optional GROQ_API_KEY
```

### 4. Run the Development Server
```bash
python -m chatbot.app
```
Access the application at [http://127.0.0.1:5000](http://127.0.0.1:5000).

For comprehensive Windows 11 setup details and troubleshooting, see [WINDOWS_STARTUP_GUIDE.md](WINDOWS_STARTUP_GUIDE.md).

---

## 🧪 Automated Testing

NeuroScan AI includes an exhaustive test suite verifying model inference, security headers, database integrity, and report generation:

```bash
# Run all automated test suites
pytest

# Run tests with code coverage metrics
pytest --cov=chatbot -v
```

---

## 📜 Clinical & Ethical Disclaimer

> **IMPORTANT MEDICAL NOTICE:**
> NeuroScan AI is an artificial intelligence research platform and clinical decision-support prototype. It is **NOT** a certified medical diagnostic device and is not intended to replace professional neurological judgment, formal cognitive screening (MMSE, MoCA), or clinical radiologist interpretations. All diagnostic outputs must be evaluated by a certified physician.

For complete clinical and regulatory information, see [docs/CLINICAL_DISCLAIMER.md](docs/CLINICAL_DISCLAIMER.md).

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
