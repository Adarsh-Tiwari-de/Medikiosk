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
| `GET` | `/api/samples` | Lists available demo prescription samples |
| `GET` | `/api/sample/{filename}` | Serves a single demo sample image |
| `POST` | `/api/scan-document` | Single-document extraction (image or PDF) |
| `POST` | `/api/scan-documents` | Batch multi-file scan (up to 10 files) |
| `POST` | `/api/merge-scans` | Lightweight in-memory merger for incremental captures |

---

## Demo Samples

Demo test images are optional and excluded from version control to prevent inadvertent disclosure of Protected Health Information (PHI). If testing locally with your own de-identified test documents, create an `ocr/samples/` directory and place test files there.

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

## Clinical Principles

> MediKiosk prepares and structures clinical data for human review.
> It **never** acts as an autonomous diagnostic engine, never recommends medication changes, and never overrides the attending clinician.
> All extracted data is presented as requiring clinician verification.

---

## Integration with Other Members

- **Member 1 (Next.js Kiosk UI):** Can call `POST /api/scan-document` to submit scanned images.
- **Member 5 (Backend / Firestore):** Receives structured JSON from this service for persistent session storage.
- **Member 2 (Doctor Dashboard):** Displays extracted and verified medication schedules and lab values.

The OCR service runs independently on port `8000` (or configurable). No database or external services are required beyond the Gemini API.
