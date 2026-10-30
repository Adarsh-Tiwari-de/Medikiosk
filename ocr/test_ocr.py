"""
Unit & Integration Tests for MediKiosk OCR Engine (Offline / Local Verification)
Does NOT invoke paid APIs or transmit medical data.
"""

import sys
import os
import io
import unittest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

# Ensure ocr/ directory is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import main
from main import (
    app,
    ALLOWED_SAMPLE_FILES,
    DEFAULT_ALLOWED_ORIGINS,
    validate_model_output,
    merge_clinical_documents,
    check_rate_limit,
    _request_history,
    _rate_limit_lock
)

client = TestClient(app)


def make_dummy_jpeg() -> bytes:
    """Creates a minimal valid JPEG image in memory for testing."""
    img = Image.new("RGB", (20, 20), color="white")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestOcrSamplesSecurity(unittest.TestCase):
    """Verifies that decommissioned sensitive samples are completely inaccessible."""

    def test_samples_allowlist_is_empty(self):
        """Allowlist must not contain previously decommissioned sensitive files."""
        self.assertEqual(len(ALLOWED_SAMPLE_FILES), 0)
        self.assertNotIn("ReportTest1.jpeg", ALLOWED_SAMPLE_FILES)
        self.assertNotIn("ReportTest02.jpeg", ALLOWED_SAMPLE_FILES)
        self.assertNotIn("FaulTEst02.jpeg", ALLOWED_SAMPLE_FILES)

    def test_get_samples_returns_empty_list(self):
        """GET /api/samples must return an empty list []."""
        resp = client.get("/api/samples")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), [])

    def test_decommissioned_samples_return_404(self):
        """GET /api/sample/{filename} must return 404 for all decommissioned filenames."""
        decommissioned = [
            "ReportTest1.jpeg",
            "ReportTest02.jpeg",
            "FaulTEst02.jpeg",
            "nonexistent.jpeg",
            "test.png"
        ]
        for name in decommissioned:
            resp = client.get(f"/api/sample/{name}")
            self.assertEqual(
                resp.status_code, 404,
                f"Expected 404 for sample '{name}', got {resp.status_code}"
            )


class TestClinicalOutputValidation(unittest.TestCase):
    """Verifies Gemini output schema validation and uncertainty flagging."""

    def test_non_dict_input_returns_safe_fallback(self):
        """Non-dictionary inputs must safely return a manual-review fallback."""
        for bad_input in [None, "invalid json string", [1, 2, 3], 42]:
            result = validate_model_output(bad_input)
            self.assertIsInstance(result, dict)
            self.assertFalse(result["isValidMedicalDocument"])
            self.assertTrue(result["needsManualReview"])
            self.assertEqual(result["clinicalVerificationStatus"], "Pending Clinician Review")

    def test_missing_or_unreadable_medication_fields_trigger_manual_review(self):
        """Unreadable or unclear dosage/frequency must flag needsManualReview: True."""
        sample_raw = {
            "isValidMedicalDocument": True,
            "documentType": "Prescription",
            "confidence": "Moderate",
            "extractedMedications": [
                {
                    "name": "Amoxicillin",
                    "dosage": "500mg",
                    "frequency": "Unclear handwriting",
                    "timing": "After meals",
                    "duration": "5 days"
                }
            ],
            "extractedLabValues": [],
            "pastConditionsIdentified": [],
            "doctorNotes": "Patient has mild pharyngitis",
            "needsManualReview": False
        }
        result = validate_model_output(sample_raw)
        self.assertTrue(result["isValidMedicalDocument"])
        self.assertTrue(
            result["needsManualReview"],
            "Unclear medication frequency must force needsManualReview=True"
        )
        self.assertEqual(result["clinicalVerificationStatus"], "Pending Clinician Review")

    def test_fully_legible_record_preserves_clean_state(self):
        """Fully legible documents with High confidence retain clean status."""
        sample_raw = {
            "isValidMedicalDocument": True,
            "documentType": "Prescription",
            "confidence": "High",
            "extractedMedications": [
                {
                    "name": "Paracetamol",
                    "dosage": "500mg",
                    "frequency": "1-0-1",
                    "timing": "After food",
                    "duration": "3 days"
                }
            ],
            "extractedLabValues": [],
            "pastConditionsIdentified": ["Fever"],
            "doctorNotes": "Hydration advised",
            "needsManualReview": False
        }
        result = validate_model_output(sample_raw)
        self.assertTrue(result["isValidMedicalDocument"])
        self.assertFalse(result["needsManualReview"])
        self.assertEqual(result["confidence"], "High")
        self.assertEqual(result["clinicalVerificationStatus"], "Pending Clinician Review")

    def test_unclear_lab_value_triggers_manual_review(self):
        """Unclear lab test value must force needsManualReview: True."""
        sample_raw = {
            "isValidMedicalDocument": True,
            "documentType": "Lab Report",
            "confidence": "High",
            "extractedMedications": [],
            "extractedLabValues": [
                {
                    "test": "HbA1c",
                    "value": "Unreadable smudge",
                    "unit": "%",
                    "status": "Unclear"
                }
            ],
            "pastConditionsIdentified": [],
            "doctorNotes": "",
            "needsManualReview": False
        }
        result = validate_model_output(sample_raw)
        self.assertTrue(result["needsManualReview"])

    def test_is_valid_medical_document_string_false_rejected(self):
        """String 'false' must NOT be evaluated as truthy; must set isValidMedicalDocument=False and needsManualReview=True."""
        sample_raw = {
            "isValidMedicalDocument": "false",
            "documentType": "Prescription",
            "confidence": "High",
            "extractedMedications": [],
            "extractedLabValues": [],
            "pastConditionsIdentified": [],
            "doctorNotes": "Some notes",
            "needsManualReview": False
        }
        result = validate_model_output(sample_raw)
        self.assertFalse(result["isValidMedicalDocument"])
        self.assertTrue(result["needsManualReview"])

    def test_is_valid_medical_document_string_true_rejected(self):
        """String 'true' is not an actual JSON boolean; must set isValidMedicalDocument=False and needsManualReview=True."""
        sample_raw = {
            "isValidMedicalDocument": "true",
            "documentType": "Prescription",
            "confidence": "High",
            "extractedMedications": [],
            "extractedLabValues": [],
            "pastConditionsIdentified": [],
            "doctorNotes": "Some notes",
            "needsManualReview": False
        }
        result = validate_model_output(sample_raw)
        self.assertFalse(result["isValidMedicalDocument"])
        self.assertTrue(result["needsManualReview"])

    def test_missing_is_valid_field_rejected(self):
        """Missing 'isValidMedicalDocument' must default to False and trigger manual review."""
        sample_raw = {
            "documentType": "Prescription",
            "confidence": "High",
            "extractedMedications": [],
            "extractedLabValues": [],
            "pastConditionsIdentified": [],
            "doctorNotes": "Notes"
        }
        result = validate_model_output(sample_raw)
        self.assertFalse(result["isValidMedicalDocument"])
        self.assertTrue(result["needsManualReview"])

    def test_missing_required_list_fields_trigger_manual_review(self):
        """Missing required list fields must force needsManualReview=True and default to empty lists."""
        sample_raw = {
            "isValidMedicalDocument": True,
            "documentType": "Prescription",
            "confidence": "High",
            "doctorNotes": "Notes"
            # extractedMedications, extractedLabValues, pastConditionsIdentified omitted
        }
        result = validate_model_output(sample_raw)
        self.assertTrue(result["needsManualReview"])
        self.assertIsInstance(result["extractedMedications"], list)
        self.assertIsInstance(result["extractedLabValues"], list)
        self.assertIsInstance(result["pastConditionsIdentified"], list)

    def test_invalid_list_types_trigger_manual_review(self):
        """Non-list types for medications, labs, or conditions must force needsManualReview=True."""
        sample_raw = {
            "isValidMedicalDocument": True,
            "documentType": "Prescription",
            "confidence": "High",
            "extractedMedications": "None prescribed",  # invalid: string instead of list
            "extractedLabValues": {"test": "sugar"},    # invalid: dict instead of list
            "pastConditionsIdentified": "Hypertension", # invalid: string instead of list
            "doctorNotes": "Routine follow up",
            "needsManualReview": False
        }
        result = validate_model_output(sample_raw)
        self.assertTrue(result["needsManualReview"])
        self.assertEqual(result["extractedMedications"], [])
        self.assertEqual(result["extractedLabValues"], [])
        self.assertEqual(result["pastConditionsIdentified"], [])

    def test_normal_valid_response(self):
        """A properly formatted, fully typed JSON response preserves valid state without manual review."""
        sample_raw = {
            "isValidMedicalDocument": True,
            "rejectionReason": None,
            "documentType": "Prescription",
            "hospitalName": "Apollo Clinic",
            "department": "Cardiology",
            "confidence": "High",
            "extractedMedications": [
                {
                    "name": "Metoprolol Succinate",
                    "dosage": "25mg",
                    "frequency": "OD",
                    "timing": "Morning after breakfast",
                    "duration": "30 days"
                }
            ],
            "extractedLabValues": [
                {
                    "test": "Blood Pressure",
                    "value": "120/80",
                    "unit": "mmHg",
                    "status": "Normal"
                }
            ],
            "pastConditionsIdentified": ["Hypertension"],
            "doctorNotes": "Patient is clinically stable. Continue current prescription.",
            "needsManualReview": False
        }
        result = validate_model_output(sample_raw)
        self.assertTrue(result["isValidMedicalDocument"])
        self.assertFalse(result["needsManualReview"])
        self.assertEqual(result["confidence"], "High")
        self.assertEqual(len(result["extractedMedications"]), 1)
        self.assertEqual(result["extractedMedications"][0]["name"], "Metoprolol Succinate")
        self.assertEqual(len(result["extractedLabValues"]), 1)
        self.assertEqual(len(result["pastConditionsIdentified"]), 1)


class TestNetworkAndCorsDefaults(unittest.TestCase):
    """Verifies restricted network bindings and localhost-only CORS defaults."""

    def test_default_origins_are_localhost_only(self):
        """Default allowed origins must restrict to loopback/localhost ports."""
        origins = [o.strip() for o in DEFAULT_ALLOWED_ORIGINS.split(",") if o.strip()]
        self.assertNotIn("*", origins)
        self.assertIn("http://localhost:3000", origins)
        self.assertIn("http://127.0.0.1:3000", origins)
        self.assertIn("http://localhost:8000", origins)
        self.assertIn("http://127.0.0.1:8000", origins)

    def test_health_check_returns_online(self):
        """GET / confirms service status without requiring external API calls."""
        resp = client.get("/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["service"], "MediKiosk OCR Engine")
        self.assertEqual(data["status"], "online")


class TestClinicalMerger(unittest.TestCase):
    """Verifies deterministic multi-document clinical record merger."""

    def test_empty_document_list_returns_safe_failure(self):
        """Merging empty document list safely reports invalid capture."""
        result = merge_clinical_documents([])
        self.assertFalse(result["isValidMedicalDocument"])
        self.assertTrue(result["needsManualReview"])
        self.assertEqual(result["clinicalVerificationStatus"], "Pending Clinician Review")

    def test_provenance_tagging_on_valid_documents(self):
        """Merged items must retain source document index."""
        doc1 = {
            "docIndex": 1,
            "filename": "slip1.jpg",
            "data": {
                "isValidMedicalDocument": True,
                "hospitalName": "Hospital A",
                "department": "Skin OPD",
                "confidence": "High",
                "extractedMedications": [{"name": "Ketoconazole 2%"}],
                "extractedLabValues": [],
                "pastConditionsIdentified": [],
                "doctorNotes": "Notes 1",
                "needsManualReview": False
            }
        }
        result = merge_clinical_documents([doc1])
        self.assertTrue(result["isValidMedicalDocument"])
        self.assertEqual(len(result["extractedMedications"]), 1)
        self.assertEqual(result["extractedMedications"][0]["sourceDocIndex"], 1)
        self.assertEqual(result["clinicalVerificationStatus"], "Pending Clinician Review")


class TestSecurityAndValidation(unittest.TestCase):
    """Verifies security boundaries, input sanity, and rate limiting."""

    def test_invalid_pdf_magic_bytes_rejected(self):
        """A file with .pdf extension or application/pdf MIME must start with %PDF- header."""
        fake_pdf = io.BytesIO(b"This is not a real PDF file header")
        resp = client.post(
            "/api/scan-document",
            files={"file": ("fake_report.pdf", fake_pdf, "application/pdf")}
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("not a valid PDF document", resp.json()["detail"])

    def test_corrupt_or_invalid_image_rejected(self):
        """Non-image binary data uploaded with image MIME type must be rejected."""
        fake_img = io.BytesIO(b"Not a valid JPEG or PNG header data stream")
        resp = client.post(
            "/api/scan-document",
            files={"file": ("corrupt.jpg", fake_img, "image/jpeg")}
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("corrupted or not a valid image format", resp.json()["detail"])

    def test_missing_gemini_api_key_returns_500(self):
        """Missing GEMINI_API_KEY environment variable raises HTTP 500 without crashing."""
        dummy_jpeg = io.BytesIO(make_dummy_jpeg())
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
            resp = client.post(
                "/api/scan-document",
                files={"file": ("test.jpg", dummy_jpeg, "image/jpeg")}
            )
            self.assertEqual(resp.status_code, 500)
            self.assertIn("GEMINI_API_KEY is missing", resp.json()["detail"])

    def test_rate_limiting_exceeded_and_retry_after_header(self):
        """Exceeding request rate limit produces HTTP 429 with Retry-After header."""
        test_ip = "192.0.2.42"
        with _rate_limit_lock:
            _request_history[test_ip] = []

        # Send up to the limit
        for _ in range(main.RATE_LIMIT_MAX_REQUESTS):
            check_rate_limit(test_ip)

        # The (MAX + 1)-th call must raise HTTPException(429)
        with self.assertRaises(HTTPException) as ctx:
            check_rate_limit(test_ip)

        self.assertEqual(ctx.exception.status_code, 429)
        self.assertIn("Retry-After", ctx.exception.headers)
        self.assertTrue(int(ctx.exception.headers["Retry-After"]) > 0)


class TestBatchSizeLimitsAndModelIsolation(unittest.TestCase):
    """Verifies that individual and aggregate batch size limits abort before calling Gemini."""

    def test_oversized_single_file_rejected_before_gemini_call(self):
        """Single file exceeding MAX_FILE_SIZE returns 413 and never calls Gemini."""
        with patch.object(main, "MAX_FILE_SIZE", 200), \
             patch.object(main, "get_gemini_client") as mock_client:
            payload = io.BytesIO(b"A" * 250)
            resp = client.post(
                "/api/scan-document",
                files={"file": ("oversized.jpg", payload, "image/jpeg")}
            )
            self.assertEqual(resp.status_code, 413)
            self.assertIn("File size exceeds maximum limit", resp.json()["detail"])
            mock_client.assert_not_called()

    def test_oversized_aggregate_batch_rejected_before_gemini_call(self):
        """Aggregate batch size exceeding MAX_BATCH_SIZE returns 413 and never calls Gemini."""
        with patch.object(main, "MAX_FILE_SIZE", 500), \
             patch.object(main, "MAX_BATCH_SIZE", 800), \
             patch.object(main, "get_gemini_client") as mock_client:
            # Two valid PDFs of 450 bytes each (total 900 bytes > 800 bytes)
            pdf1 = b"%PDF-1.4 " + b"A" * 441
            pdf2 = b"%PDF-1.4 " + b"B" * 441
            files = [
                ("files", ("f1.pdf", io.BytesIO(pdf1), "application/pdf")),
                ("files", ("f2.pdf", io.BytesIO(pdf2), "application/pdf"))
            ]
            resp = client.post("/api/scan-documents", files=files)
            self.assertEqual(resp.status_code, 413)
            self.assertIn("Aggregate batch size exceeds maximum limit", resp.json()["detail"])
            mock_client.assert_not_called()

    def test_oversized_individual_file_in_batch_rejected_before_gemini_call(self):
        """An individual file exceeding limit in a batch scan returns 413 and never calls Gemini."""
        with patch.object(main, "MAX_FILE_SIZE", 500), \
             patch.object(main, "MAX_BATCH_SIZE", 2000), \
             patch.object(main, "get_gemini_client") as mock_client:
            pdf_big = b"%PDF-1.4 " + b"A" * 591  # 600 bytes > 500 limit
            pdf_normal = b"%PDF-1.4 " + b"B" * 91 # 100 bytes
            files = [
                ("files", ("f1_too_big.pdf", io.BytesIO(pdf_big), "application/pdf")),
                ("files", ("f2_normal.pdf", io.BytesIO(pdf_normal), "application/pdf"))
            ]
            resp = client.post("/api/scan-documents", files=files)
            self.assertEqual(resp.status_code, 413)
            self.assertIn("File #1 exceeds maximum limit", resp.json()["detail"])
            mock_client.assert_not_called()


class TestClinicalUncertaintyMarkers(unittest.TestCase):
    """Verifies targeted uncertainty detection without naive global blocklists."""

    def test_uncertainty_in_medications_triggers_manual_review(self):
        """Dosage/frequency containing 'likely', 'approximate', or 'suspected' forces manual review."""
        markers = ["likely 500mg", "approx 5 days", "suspected amoxicillin", "probably BD"]
        for marker in markers:
            raw = {
                "isValidMedicalDocument": True,
                "documentType": "Prescription",
                "confidence": "Moderate",
                "extractedMedications": [
                    {
                        "name": "Medication X",
                        "dosage": marker,
                        "frequency": "OD",
                        "timing": "After food",
                        "duration": "5 days"
                    }
                ],
                "extractedLabValues": [],
                "pastConditionsIdentified": ["Hypertension"],
                "doctorNotes": "Routine consultation",
                "needsManualReview": False
            }
            result = validate_model_output(raw)
            self.assertTrue(
                result["needsManualReview"],
                f"Marker '{marker}' in medications must set needsManualReview=True"
            )

    def test_uncertainty_in_lab_values_triggers_manual_review(self):
        """Lab values marked 'borderline', 'approximate', or 'unclear' force manual review."""
        markers = ["borderline", "approx 140", "doubtful"]
        for marker in markers:
            raw = {
                "isValidMedicalDocument": True,
                "documentType": "Lab Report",
                "confidence": "Moderate",
                "extractedMedications": [],
                "extractedLabValues": [
                    {
                        "test": "Fasting Blood Sugar",
                        "value": marker,
                        "unit": "mg/dL",
                        "status": "Borderline"
                    }
                ],
                "pastConditionsIdentified": [],
                "doctorNotes": "",
                "needsManualReview": False
            }
            result = validate_model_output(raw)
            self.assertTrue(
                result["needsManualReview"],
                f"Marker '{marker}' in lab values must set needsManualReview=True"
            )

    def test_uncertainty_in_conditions_triggers_manual_review(self):
        """Provisional/suspected conditions force manual review."""
        provisional_conditions = [
            "Suspected Dengue",
            "Possible Appendicitis",
            "Rule out TB",
            "Probable Asthma"
        ]
        for cond in provisional_conditions:
            raw = {
                "isValidMedicalDocument": True,
                "documentType": "Prescription",
                "confidence": "High",
                "extractedMedications": [],
                "extractedLabValues": [],
                "pastConditionsIdentified": [cond],
                "doctorNotes": "Patient examination normal.",
                "needsManualReview": False
            }
            result = validate_model_output(raw)
            self.assertTrue(
                result["needsManualReview"],
                f"Provisional condition '{cond}' must set needsManualReview=True"
            )

    def test_ordinary_explanatory_notes_do_not_falsely_flag_review(self):
        """Conversational text in doctorNotes (e.g. follow-up timing) does not falsely trigger manual review."""
        raw = {
            "isValidMedicalDocument": True,
            "documentType": "Prescription",
            "confidence": "High",
            "extractedMedications": [
                {
                    "name": "Paracetamol",
                    "dosage": "500mg",
                    "frequency": "1-0-1",
                    "timing": "After meals",
                    "duration": "3 days"
                }
            ],
            "extractedLabValues": [],
            "pastConditionsIdentified": ["Hypertension"],
            "doctorNotes": "Patient is likely to follow up next Tuesday with report. Estimated return in 2 weeks.",
            "needsManualReview": False
        }
        result = validate_model_output(raw)
        self.assertTrue(result["isValidMedicalDocument"])
        self.assertFalse(
            result["needsManualReview"],
            "Ordinary conversational explanatory notes must NOT falsely trigger manual review."
        )

    def test_explicit_illegible_notes_do_flag_review(self):
        """Explicit notes stating illegible handwriting do trigger manual review."""
        raw = {
            "isValidMedicalDocument": True,
            "documentType": "Prescription",
            "confidence": "High",
            "extractedMedications": [],
            "extractedLabValues": [],
            "pastConditionsIdentified": [],
            "doctorNotes": "Unclear handwriting on prescription slip. Partially unreadable notes.",
            "needsManualReview": False
        }
        result = validate_model_output(raw)
        self.assertTrue(result["needsManualReview"])


if __name__ == "__main__":
    unittest.main()
