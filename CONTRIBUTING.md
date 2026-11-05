# Contributing to NeuroScan AI

Thank you for contributing to the NeuroScan AI Alzheimer's Detection & Clinical Diagnosis Platform!

## Code of Conduct

We are committed to maintaining a collaborative, respectful, and inclusive environment. Please communicate constructively and prioritize patient privacy and clinical integrity in all contributions.

## Development Workflow

1. **Fork the Repository**:
   Fork the repository to your own GitHub account, then clone it to your local machine:
   ```bash
   git clone https://github.com/<your-username>/NeuroScan-AI-Alzheimer-s-Detection-Diagnosis-Platform.git
   cd NeuroScan-AI-Alzheimer-s-Detection-Diagnosis-Platform
   ```

2. **Create a Topic Branch**:
   Create an isolated branch off `main` with a descriptive name:
   ```bash
   git checkout -b feat/your-feature-name
   ```

3. **Virtual Environment Setup**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # on Windows: .\.venv\Scripts\Activate.ps1
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Commit Standards

Please write clear, meaningful commit messages using standard conventional commit prefixes:

- `feat(scope)`: New capability or enhancement
- `fix(scope)`: Bug fix or security hardening
- `test(scope)`: Unit or integration test coverage
- `perf(scope)`: Performance optimization or database indexing
- `docs(scope)`: Documentation, README, or guide updates

## Testing & Verification

Before opening a pull request, verify that:
1. All Python files pass bytecode compilation without syntax errors:
   ```bash
   python -m py_compile main.py models/resnet_model.py chatbot/app.py
   ```
2. The automated test suite executes cleanly:
   ```bash
   python -m unittest discover -s tests -p "test_*.py"
   ```
3. Medical, clinical, and regulatory disclaimers remain intact and visible.

## Opening a Pull Request

1. Push your topic branch to your fork on GitHub:
   ```bash
   git push -u origin feat/your-feature-name
   ```
2. Open a Pull Request targeting `main`.
3. Fill out the pull request template with a concise summary of changes and testing evidence.
