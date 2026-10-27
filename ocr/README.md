# MediKiosk — OCR Module (Member 4)

> **Medical Document Digitization Service**  
> Part of the [MediKiosk](https://github.com/Adarsh-Tiwari-de/Medikiosk) pre-consultation platform.

This module is owned by **Member 4** and provides the medical document OCR backend. It accepts handwritten Indian OPD prescriptions, clinic slips, and multi-page PDF lab reports and returns structured clinical entities using Google Gemini Vision.

---

## Features

- **Multimodal OCR:** JPEG / PNG / WebP images and multi-page PDF documents (up to 10 MB each)
- **Non-Medical Frame Detection:** Automatically rejects selfies, rooms, and blank pages
- **Multi-Document Patient Packets:** Scan multiple slips in one session; all data is merged with source-attribution badges (`Doc #1`, `Doc #2`)
- **Bilingual UI:** Hindi + English with voice readout (Web Speech API)
- **5-Model Fallback Chain:** Automatically retries on Google Cloud quota or capacity errors
- **Security Hardened:** Allowlist-only sample serving, bounded reads, PDF magic-byte validation, EXIF correction

---

## Setup

### 1. Prerequisites

- Python 3.10+
- A [Google Gemini API key](https://aistudio.google.com/app/apikey)

### 2. Install dependencies

```bash
cd ocr/
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and paste your GEMINI_API_KEY value
```

> **Never commit `.env`** — it is excluded by the repository's `.gitignore`.

### 4. Start the service

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/` | Health check — confirms service is online |
| `GET` | `/camera` or `/dashboard` | Serves the Clinical Daylight Kiosk UI |
| `GET` | `/api/samples` | Lists available demo prescription samples (currently empty; old sensitive samples decommissioned) |
| `GET` | `/api/sample/{filename}` | Serves an approved sample image (404 for unapproved/decommissioned files) |
| `POST` | `/api/scan-document` | Single-document extraction (image or PDF) |
| `POST` | `/api/scan-documents` | Batch multi-file scan (up to 10 files) |
| `POST` | `/api/merge-scans` | Lightweight in-memory merger for incremental captures |

---

## Demo Samples Policy

To protect patient privacy and comply with healthcare data standards, all previous real-world sample images (`ReportTest1.jpeg`, `ReportTest02.jpeg`, `FaulTEst02.jpeg`) have been permanently decommissioned and excluded from version control. `GET /api/samples` returns an empty list `[]` and `GET /api/sample/{filename}` returns `404` until explicitly approved synthetic, de-identified datasets are integrated.

---

## Directory Structure

```
ocr/
├── main.py             # FastAPI backend (Member 4 OCR engine)
├── camera.html         # Clinical Daylight Kiosk UI (served at /camera)
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
└── README.md           # This file
```

---

## Network Exposure & Security Architecture

### Local Development vs Production

- **Local Development Default:** The service binds to the loopback interface (`127.0.0.1:8000`) and restricts CORS to local development ports (`http://localhost:3000`, `http://127.0.0.1:3000`, `http://localhost:8000`, `http://127.0.0.1:8000`). It is not exposed to the public internet or external networks by default.
- **Production Deployment Requirements:**
  - **Authentication & Access Control:** CORS headers are a browser-side sandbox mechanism and **do not provide authentication or authorization**. Production hospital deployment requires an upstream API Gateway or reverse proxy (e.g., NGINX, Envoy, Traefik) enforcing mutual TLS (mTLS) or OAuth2/bearer tokens.
  - **Network Isolation:** The OCR service should run inside a private clinical VLAN or internal Kubernetes cluster with no direct public ingress.
  - **Medical Data Privacy Review:** A formal Data Protection Impact Assessment (DPIA) under India's Digital Personal Data Protection (DPDP) Act 2023, DISHA, or HIPAA must be completed before handling real patient records.
  - **Generic Logging:** The service uses generic document identifiers in server logs to prevent patient names or identifiers from leaking through file names.

---

## Clinical Principles & Grounding

> MediKiosk prepares and structures clinical data for human review.
> - **Zero Fabrication:** The extraction prompt strictly prohibits deducing or guessing medications, dosages, frequencies, or lab values from medical context.
> - **Uncertainty Flagging:** Any unreadable, ambiguous, or illegible fields are explicitly marked (`Unclear` / `Unreadable`) and automatically trigger `"needsManualReview": true`.
> - **Preliminary Status:** All extracted entities carry `"clinicalVerificationStatus": "Pending Clinician Review"` and require review by a licensed healthcare professional before entering an EHR.

---

## Integration with Other Members

- **Member 1 (Next.js Kiosk UI):** Can call `POST /api/scan-document` to submit scanned images.
- **Member 5 (Backend / Firestore):** Receives structured JSON from this service for persistent session storage.
- **Member 2 (Doctor Dashboard):** Displays extracted and verified medication schedules and lab values.

The OCR service runs independently on port `8000` (or configurable). No database or external services are required beyond the Gemini API.
