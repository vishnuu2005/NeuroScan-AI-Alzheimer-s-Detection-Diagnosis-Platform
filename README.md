# NeuroScan AI (Alzheimer's MRI classifier + chatbot)

Minimal README to run the project locally and push to GitHub.

Prerequisites
- Python 3.10+ (3.13 used in dev)
- Git

Setup
1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Configuration
- Copy or edit `.env` at project root to set:
  - `GROQ_API_KEY` — your Groq API key (optional). If invalid, the app uses a local fallback.
  - `FLASK_SECRET_KEY` — optional secret for sessions.
  - `USE_GROQ` — set to `0` to disable Groq and always use local fallback.
  - `PREDICTION_THRESHOLD` — float (e.g. `0.3`) lower -> fewer "Uncertain" replies.
  - `IMAGE_SIZE` — input size for preprocessing (default 224).

Running
- Start the web app (development mode):

```powershell
Set-Location -Path 'f:/AlzeigmersChatBot/AlzeigmersChatBot/AlzeigmersChatBot'
python -m chatbot.app
```

- Open: http://127.0.0.1:5000/

Smoke tests
- A smoke-test script is available at `tools/smoke_test.py` to register/login, upload sample image, and call `/chat`.

```powershell
python tools/smoke_test.py
```

Notes and troubleshooting
- If Groq returns `Invalid API Key` (401) the app prints a warning and uses a local fallback response.
- Model checkpoints are in `models/resnet2d_alzheimers.pth`.
- If you see `ModuleNotFoundError` for local imports, run the app with `python -m chatbot.app` from the project folder so package imports resolve.

Pushing to GitHub
```bash
git init
git add .
git commit -m "Initial import: NeuroScan AI"
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```

If you want I can open a PR, add CI, or create a `.gitignore` tuned for Python (venv, caches, model checkpoints).