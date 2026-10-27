"""
Unit & Integration Tests for MediKiosk OCR Engine (Offline / Local Verification)
Does NOT invoke paid APIs or transmit medical data.
"""

import sys
import os
import unittest
from fastapi.testclient import TestClient

# Ensure ocr/ directory is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import (
    app,
    ALLOWED_SAMPLE_FILES,
    DEFAULT_ALLOWED_ORIGINS,
    validate_model_output,
    merge_clinical_documents
)

client = TestClient(app)


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


if __name__ == "__main__":
    unittest.main()
