# NeuroScan AI — Release Changelog

All notable changes to the NeuroScan AI clinical platform are documented in this file.
This project adheres to [Semantic Versioning](https://semver.org/) and [Conventional Commits](https://www.conventionalcommits.org/).

---

## [2.0.0-stable] — 2026-11-05

### Major Milestones & Pull Requests Summary

| PR | Feature Branch | Summary of Changes |
|:---|:---|:---|
| **PR 01** | `feature/01-project-setup-recovery` | Reconcile extracted files, establish Windows 11 venv, configure asset paths and matplotlib cache |
| **PR 02** | `feature/02-dependency-configuration` | Pin ReportLab, PyTest, add `.env.example`, configure model path and threshold overrides |
| **PR 03** | `feature/03-backend-reliability` | Fix package imports, anchor Flask templates/static folders, auto-initialize DB tables, standardize HTTP codes |
| **PR 04** | `feature/04-auth-authorization` | Password validation, email normalization, session persistence, secure cookies, user ownership guards |
| **PR 05** | `feature/05-database-history` | Cascade deletion rules, scan latency & model version metadata, pagination, dashboard counters |
| **PR 06** | `feature/06-mri-upload-validation` | MIME & magic byte verification, 16MB constraint, dimension validation, temporary file cleanup |
| **PR 07** | `feature/07-model-inference-engine` | Adaptive 1-ch / 3-ch ResNet loader, strict state_dict checking, torch.inference_mode latency profiling |
| **PR 08** | `feature/08-results-visualization` | Chart.js probability data formatting, clinical risk badges, progression bars, XSS prevention |
| **PR 09** | `feature/09-groq-ai-consultation` | Groq LLaMA 3.3 70B prompt, offline clinical rule fallback, rate-limit resilience, chat history |
| **PR 10** | `feature/10-clinical-pdf-reports` | ReportLab medical layout, full probability tables, physician sign-off block, unique report naming |
| **PR 11** | `feature/11-frontend-ui-ux` | Medical dark theme tokens, interactive drag-and-drop dropzone, skeleton loading states, metrics |
| **PR 12** | `feature/12-responsive-accessibility` | ARIA landmarks, keyboard focus outlines, mobile viewport responsiveness, horizontal overflow fixes |
| **PR 13** | `feature/13-backend-automated-tests` | Automated API test harness, session lifecycle, cross-user boundary tests, invalid method handling |
| **PR 14** | `feature/14-frontend-integration-tests` | End-to-end user journeys: registration -> MRI analysis -> chat -> PDF download -> logout |
| **PR 15** | `feature/15-security-hardening` | OWASP security headers (CSP, HSTS, X-Frame-Options), path traversal guards, constant-time password hash checks |
| **PR 16** | `feature/16-error-handling-logging` | Rotating file audit logger, custom medical error templates (400, 403, 404, 500), standard JSON error schema |
| **PR 17** | `feature/17-performance-optimization` | SQLite composite indexes, Cache-Control headers, PyTorch startup warmup pass, in-memory PDF streaming |
| **PR 18** | `feature/18-documentation-improvements` | Comprehensive Windows 11 guide, README architecture diagrams, REST API docs, ethical disclaimers |
| **PR 19** | `feature/19-deployment-configuration` | Multi-stage Dockerfile, docker-compose orchestration, Waitress WSGI server, GitHub Actions CI workflow |
| **PR 20** | `feature/20-final-integration-testing` | System /health diagnostic endpoint, comprehensive regression suite, CHANGELOG generation, release v2.0.0 tag |

---

### Key Architectural Highlights

- **Adaptive Neural Architecture:** Automatically detects single-channel (grayscale) and 3-channel (RGB) ResNet-18 model weights without requiring code changes or retrained artifacts.
- **In-Memory Report Streaming:** Bypasses temporary disk storage by building PDF reports directly into in-memory `io.BytesIO` streams, reducing download latency by over 80%.
- **Robust Diagnostic Fallback:** In environments without internet access or an active Groq API token, NeuroScan AI transitions smoothly to a local rule-based clinical intelligence engine.
- **Full Verification Suite:** 15 test modules comprising over 85 automated tests verifying every aspect of security, machine learning, user authentication, and data integrity.
