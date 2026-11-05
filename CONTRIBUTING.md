# Contributing to NeuroScan AI

Thank you for your interest in contributing to **NeuroScan AI**! We welcome contributions from machine learning engineers, clinical researchers, frontend developers, and security auditors.

---

## 1. Development Principles

1. **Patient Safety First:** Never alter diagnostic inference pipelines, class thresholding, or risk calculations without rigorous statistical verification and medical literature backing.
2. **Conventional Commits:** We enforce structured, semantic Git commit messages.
3. **Automated Testing:** All pull requests must pass 100% of the pytest suite and introduce new unit tests for any added features or bug fixes.
4. **Security Integrity:** Never bypass authentication boundaries, upload validations, or OWASP security headers.

---

## 2. Commit Message Standards

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat(scope): ...` — A new user-facing or backend capability.
- `fix(scope): ...` — A bug fix.
- `perf(scope): ...` — A performance improvement without functional changes.
- `test(scope): ...` — Adding or correcting test suites.
- `security(scope): ...` — Hardening defenses, input validation, or headers.
- `docs(scope): ...` — Documentation, guides, or API specifications.
- `refactor(scope): ...` — Code changes that neither fix a bug nor add a feature.
- `chore(scope): ...` — Dependency bumps, workspace adjustments, or build tasks.

---

## 3. Pull Request Guidelines

1. **Branching Model:** Create a separate feature branch from `main` or the latest staging branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Never Commit Secrets:** Ensure `.env`, SQLite databases (`*.db`), and private API keys are excluded by `.gitignore`.
3. **Run Pre-Commit Verification:**
   ```bash
   pytest -v
   ```
4. **Pull Request Descriptions:** Every PR must clearly document:
   - Problem statement / Motivation
   - Technical implementation details
   - Testing methodology and output
   - Screenshots for any user-interface modifications
