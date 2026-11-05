# NeuroScan AI — RESTful API Documentation

This specification documents the endpoints, payload contracts, authentication boundaries, and error semantics for NeuroScan AI's clinical backend.

---

## 1. Authentication & Security

All authenticated endpoints require an active session cookie (`neuroscan_session`). Cookies are managed automatically by web clients upon calling `/login`.

### Standard Security Headers
All API responses include OWASP-compliant headers:
- `Content-Security-Policy: default-src 'self' ...`
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: SAMEORIGIN`
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Cache-Control: no-cache, no-store, must-revalidate` (for API endpoints)

---

## 2. API Endpoints

### 2.1 Authentication

#### `POST /login`
Authenticates a clinician using email and password.

- **Request Body (`application/x-www-form-urlencoded`):**
  - `email` (string, required): Clinician email address.
  - `password` (string, required): Account password.
- **Responses:**
  - `302 Found`: Redirect to `/dashboard` on success.
  - `400 Bad Request`: Missing credentials.
  - `200 OK`: Renders `login.html` with flash error on failure.

#### `POST /register`
Registers a new clinician account.

- **Request Body (`application/x-www-form-urlencoded`):**
  - `name` (string, required): Full name.
  - `email` (string, required): Valid clinical email address.
  - `password` (string, required): Password between 8 and 128 characters.
- **Responses:**
  - `302 Found`: Redirects to `/login` upon creation or validation failure.

#### `GET /logout`
Terminates the current clinical session and clears session state.

---

### 2.2 Clinical Consultation

#### `POST /api/chat`
Sends a message to the AI Clinical Consultation engine (Groq LLaMA 3.3 70B or offline clinical rule engine).

- **Headers:** `Content-Type: application/json`
- **Request Body (`application/json`):**
  ```json
  {
    "message": "What clinical interventions are indicated for mild cognitive impairment?"
  }
  ```
- **Response (`200 OK`):**
  ```json
  {
    "response": "Clinical interventions for early cognitive impairment typically focus on...",
    "status": "success",
    "timestamp": "2026-11-05T08:30:00.000000Z"
  }
  ```
- **Error Responses:**
  - `400 Bad Request`: Payload is not JSON or `message` is empty.
  - `302 Found`: Unauthenticated session redirected to login.

---

### 2.3 Diagnostic Records & History

#### `GET /api/scans`
Retrieves a paginated list of recent MRI diagnostic evaluations for the authenticated clinician.

- **Headers:** `Accept: application/json`
- **Response (`200 OK`):**
  ```json
  {
    "count": 1,
    "scans": [
      {
        "id": 142,
        "prediction": "MildDemented",
        "confidence": 87.45,
        "all_probs": {
          "NonDemented": 3.12,
          "VeryMildDemented": 8.15,
          "MildDemented": 87.45,
          "ModerateDemented": 1.28
        },
        "risk_tier": "mild",
        "processing_ms": 132.4,
        "model_version": "ResNet18-3ch",
        "created_at": "2026-11-05T08:15:22.105432"
      }
    ]
  }
  ```

#### `GET /api/scans/<int:scan_id>`
Retrieves comprehensive details for a specific diagnostic scan by database ID.

- **Response (`200 OK`):**
  ```json
  {
    "id": 142,
    "prediction": "MildDemented",
    "confidence": 87.45,
    "all_probs": {
      "NonDemented": 3.12,
      "VeryMildDemented": 8.15,
      "MildDemented": 87.45,
      "ModerateDemented": 1.28
    },
    "risk_tier": "mild",
    "warning": null,
    "processing_ms": 132.4,
    "model_version": "ResNet18-3ch",
    "created_at": "2026-11-05T08:15:22.105432"
  }
  ```
- **Error Responses:**
  - `403 Forbidden`: Authenticated user is not the owner of the requested scan.
  - `404 Not Found`: Scan record ID does not exist.

---

### 2.4 Diagnostic Reports

#### `GET /download_report`
Generates and streams a PDF diagnostic report for the most recent in-session scan.

- **Response (`200 OK`):**
  - `Content-Type: application/pdf`
  - `Content-Disposition: attachment; filename=neuroscan_report_MildDemented.pdf`
- **Error Responses:**
  - `400 Bad Request`: No scan performed in active session.

#### `GET /download_report/<int:scan_id>`
Streams a PDF diagnostic report for a specific archived scan ID.

- **Response (`200 OK`):**
  - `Content-Type: application/pdf`
  - `Content-Disposition: attachment; filename=neuroscan_report_scan_142.pdf`
- **Error Responses:**
  - `403 Forbidden`: Unauthorized scan access.
  - `404 Not Found`: Scan ID does not exist.

---

## 3. Standard JSON Error Schema

On all `/api/*` routes and requests with `Accept: application/json`, errors are returned in standard RFC 7807-inspired JSON:

```json
{
  "error": "Error description message",
  "status": 400,
  "timestamp": "2026-11-05T08:35:12.456789Z",
  "details": "Optional diagnostic details"
}
```
