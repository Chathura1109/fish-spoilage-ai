"""Regression tests for the trained model artifact and prediction API."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from predict_api import app


BASE_DIR = Path(__file__).resolve().parent
client = TestClient(app)
TEST_TOKEN = "test-ai-service-token"


@pytest.fixture(autouse=True)
def configure_service_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every API test a deterministic service token."""
    monkeypatch.setenv("AI_SERVICE_TOKEN", TEST_TOKEN)


def auth_headers() -> dict[str, str]:
    """Return the valid bearer token used by prediction requests."""
    return {"Authorization": f"Bearer {TEST_TOKEN}"}


def valid_payload() -> dict:
    """Return one physically consistent request shared by multiple tests."""
    return {
        "fishSpecies": "Yellowfin Tuna",
        "currentProductTemperature": 3.5,
        "averageProductTemperature": 3.2,
        "minimumProductTemperature": 2.8,
        "maximumProductTemperature": 5.1,
        "airTemperature": 4.0,
        "humidity": 81,
        "storageDurationHours": 36,
        "transportDurationHours": 4,
        "timeAboveLimitMinutes": 12,
        "temperatureViolationCount": 12,
        "timeSinceCatchHours": 36,
    }


def test_health_check_identifies_prototype() -> None:
    """The health response must disclose the synthetic decision-support role."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["decisionSupportOnly"] is True
    assert response.json()["datasetType"] == "synthetic prototype"


def test_valid_prediction_and_case_insensitive_species() -> None:
    """A normal request must return the complete public response contract."""
    payload = valid_payload()
    payload["fishSpecies"] = "  yellowfin tuna  "
    response = client.post("/predict", json=payload, headers=auth_headers())
    body = response.json()
    assert response.status_code == 200
    assert body["riskLevel"] in {"LOW", "MEDIUM", "HIGH"}
    assert 0 <= body["confidence"] <= 1
    assert set(body) == {
        "riskLevel",
        "confidence",
        "probabilities",
        "recommendation",
        "modelVersion",
    }
    assert set(body["probabilities"]) == {"LOW", "MEDIUM", "HIGH"}
    assert sum(body["probabilities"].values()) == pytest.approx(1.0)
    assert body["confidence"] == body["probabilities"][body["riskLevel"]]
    assert body["modelVersion"] == "spoilage-v1"


@pytest.mark.parametrize(
    ("changes", "expected_message"),
    [
        ({"storageDurationHours": -1}, "greater than or equal to 0"),
        ({"humidity": 150}, "less than or equal to 100"),
        (
            {"currentProductTemperature": 12, "maximumProductTemperature": 10},
            "currentProductTemperature must be between",
        ),
        (
            {"averageProductTemperature": 12, "maximumProductTemperature": 10},
            "averageProductTemperature must be between",
        ),
        (
            {"storageDurationHours": 4, "transportDurationHours": 5},
            "transportDurationHours cannot exceed storageDurationHours",
        ),
        (
            {"storageDurationHours": 40, "timeSinceCatchHours": 36},
            "storageDurationHours cannot exceed timeSinceCatchHours",
        ),
        (
            {"timeAboveLimitMinutes": 3000},
            "timeAboveLimitMinutes cannot exceed the total storage duration",
        ),
    ],
)
def test_invalid_measurements_are_rejected(
    changes: dict, expected_message: str
) -> None:
    """Each invalid physical condition should produce a helpful HTTP 422."""
    payload = valid_payload()
    payload.update(changes)
    response = client.post("/predict", json=payload, headers=auth_headers())
    assert response.status_code == 422
    assert expected_message in response.text


def test_unknown_species_returns_clear_client_error() -> None:
    """Unsupported species should fail clearly instead of being guessed."""
    payload = valid_payload()
    payload["fishSpecies"] = "Unknown fish"
    response = client.post("/predict", json=payload, headers=auth_headers())
    assert response.status_code == 400
    assert "Known species" in response.json()["detail"]


def test_extra_fields_are_rejected() -> None:
    """Unexpected JSON keys often indicate an integration naming mistake."""
    payload = valid_payload()
    payload["secretUnexpectedValue"] = 123
    response = client.post("/predict", json=payload, headers=auth_headers())
    assert response.status_code == 422


@pytest.mark.parametrize(
    "headers",
    [None, {"Authorization": "Bearer wrong-token"}],
)
def test_prediction_requires_valid_bearer_token(
    headers: dict[str, str] | None,
) -> None:
    """Missing and incorrect service credentials must be rejected."""
    response = client.post("/predict", json=valid_payload(), headers=headers)
    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}


def test_training_report_has_no_batch_leakage() -> None:
    """Protect the two most important saved-model evaluation requirements."""
    metrics = json.loads((BASE_DIR / "model_metrics.json").read_text(encoding="utf-8"))
    assert metrics["dataset_is_synthetic"] is True
    assert metrics["decision_support_only"] is True
    assert metrics["batch_overlap_count"] == 0
    assert metrics["classification_report"]["HIGH"]["recall"] >= 0.80
