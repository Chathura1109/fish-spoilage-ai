"""Expose the trained spoilage model through a validated REST API.

Run with:
    uvicorn predict_api:app --reload --port 8000

Interactive documentation is then available at http://127.0.0.1:8000/docs.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Load the versioned model artifact
# ---------------------------------------------------------------------------
# Resolve from this file so Uvicorn can be launched from another directory.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "spoilage_model.joblib"
DECISION_SUPPORT_NOTICE = (
    "Prototype decision support only; not a food-safety certification. "
    "Confirm decisions through an authorised quality inspection."
)


def load_model_bundle(path: Path) -> dict:
    """Load the artifact and verify that it contains all required metadata."""
    try:
        loaded = joblib.load(path)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"{path.name} not found. Run train_model.py before starting the API."
        ) from error
    except Exception as error:
        raise RuntimeError(f"Could not load {path.name}: {error}") from error

    required_keys = {
        "model",
        "feature_columns",
        "known_species",
        "model_version",
        "dataset_is_synthetic",
        "decision_support_only",
    }
    missing_keys = sorted(required_keys - set(loaded))
    if missing_keys:
        raise RuntimeError(
            "The saved model uses an outdated or incomplete format. "
            f"Missing keys: {missing_keys}. Run train_model.py again."
        )
    return loaded


bundle = load_model_bundle(MODEL_PATH)
model = bundle["model"]
feature_columns = bundle["feature_columns"]
known_species = bundle["known_species"]
model_version = bundle["model_version"]
# `casefold` lets clients use different letter casing without changing the
# canonical value used during training.
species_lookup = {species.casefold(): species for species in known_species}
species_aliases = {
    # The synthetic CSV contains the broader class "Tuna". Keep the mapping
    # explicit until real Yellowfin Tuna samples are added to the dataset.
    "yellowfin tuna": "Tuna",
}

app = FastAPI(
    title="Fish Spoilage Risk Prediction API",
    version=model_version,
    description=DECISION_SUPPORT_NOTICE,
)


# ---------------------------------------------------------------------------
# Request and response schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """Telemetry accepted from the Spring Boot backend or another client."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
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
                "hasTemperatureTelemetry": True,
                "temperatureReadingCount": 48,
            }
        },
    )

    fishSpecies: str = Field(min_length=1, max_length=100)
    currentProductTemperature: float = Field(ge=-5, le=40)
    averageProductTemperature: float = Field(ge=-5, le=40)
    minimumProductTemperature: float = Field(ge=-10, le=40)
    maximumProductTemperature: float = Field(ge=-5, le=50)
    airTemperature: float = Field(ge=-10, le=50)
    storageDurationHours: float = Field(ge=0, le=8760)
    transportDurationHours: float = Field(ge=0, le=720)
    humidity: float = Field(ge=0, le=100)
    timeAboveLimitMinutes: float = Field(ge=0, le=525_600)
    temperatureViolationCount: int = Field(ge=0, le=10_000)
    timeSinceCatchHours: float = Field(ge=0, le=8760)
    hasTemperatureTelemetry: bool | None = None
    temperatureReadingCount: int | None = None

    @model_validator(mode="after")
    def validate_physical_relationships(self) -> "PredictRequest":
        """Reject fields that are valid alone but contradictory together."""
        if not (
            self.minimumProductTemperature
            <= self.currentProductTemperature
            <= self.maximumProductTemperature
        ):
            raise ValueError(
                "currentProductTemperature must be between "
                "minimumProductTemperature and maximumProductTemperature"
            )
        if not (
            self.minimumProductTemperature
            <= self.averageProductTemperature
            <= self.maximumProductTemperature
        ):
            raise ValueError(
                "averageProductTemperature must be between "
                "minimumProductTemperature and maximumProductTemperature"
            )
        if self.transportDurationHours > self.storageDurationHours:
            raise ValueError(
                "transportDurationHours cannot exceed storageDurationHours"
            )
        if self.storageDurationHours > self.timeSinceCatchHours:
            raise ValueError("storageDurationHours cannot exceed timeSinceCatchHours")
        if self.timeAboveLimitMinutes > self.storageDurationHours * 60:
            raise ValueError(
                "timeAboveLimitMinutes cannot exceed the total storage duration"
            )
        return self


class RiskProbabilities(BaseModel):
    """Calibrated probability distribution over all supported classes."""

    LOW: float = Field(ge=0, le=1)
    MEDIUM: float = Field(ge=0, le=1)
    HIGH: float = Field(ge=0, le=1)


class PredictResponse(BaseModel):
    """Stable public response contract used by backend and mobile clients."""

    riskLevel: Literal["LOW", "MEDIUM", "HIGH"]
    confidence: float = Field(ge=0, le=1)
    probabilities: RiskProbabilities
    recommendation: str
    modelVersion: str


def build_recommendation(risk_level: str) -> str:
    """Convert the predicted class into a short operational recommendation."""
    if risk_level == "HIGH":
        return "Isolate the batch, maintain temperature below 4\u00b0C, and inspect immediately."
    if risk_level == "MEDIUM":
        return "Inspect the batch and maintain temperature below 4\u00b0C."
    return "Continue monitoring and maintain temperature below 4\u00b0C."


@app.get("/")
def health_check() -> dict:
    """Return lightweight service metadata for uptime checks."""
    return {
        "status": "AI service running",
        "modelVersion": model_version,
        "datasetType": "synthetic prototype",
        "decisionSupportOnly": True,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(
    request: PredictRequest,
    authorization: str | None = Header(default=None),
) -> PredictResponse:
    """Validate telemetry, run the model, and format the prediction."""
    expected_authorization = f"Bearer {os.environ['AI_SERVICE_TOKEN']}"
    if authorization is None or not secrets.compare_digest(
        authorization, expected_authorization
    ):
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Resolve exact names and supported aliases to a species seen in training.
    requested_species = request.fishSpecies.casefold()
    canonical_species = species_lookup.get(requested_species)
    if canonical_species is None:
        canonical_species = species_aliases.get(requested_species)
    if canonical_species is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown fishSpecies '{request.fishSpecies}'. "
                f"Known species: {known_species}"
            ),
        )

    # Six request fields are not present in the current training CSV:
    # minimumProductTemperature, airTemperature, timeAboveLimitMinutes, and
    # timeSinceCatchHours, hasTemperatureTelemetry, and temperatureReadingCount.
    # They are accepted or validated above, but the model cannot use them until
    # real columns are collected and FEATURE_COLUMNS is expanded.
    row = pd.DataFrame(
        [
            {
                "fish_species": canonical_species,
                "current_temperature": request.currentProductTemperature,
                "average_temperature": request.averageProductTemperature,
                "maximum_temperature": request.maximumProductTemperature,
                "storage_duration_hours": request.storageDurationHours,
                "humidity": request.humidity,
                "transport_duration_hours": request.transportDurationHours,
                "temperature_violation_count": request.temperatureViolationCount,
            }
        ]
    )[feature_columns]

    # Request both the predicted class and calibrated class probabilities.
    risk_level = str(model.predict(row)[0])
    predicted_probabilities = model.predict_proba(row)[0]
    probability_by_class = {
        label: float(predicted_probabilities[list(model.classes_).index(label)])
        for label in ("LOW", "MEDIUM", "HIGH")
    }
    rounded_probabilities = {
        label: round(probability, 2)
        for label, probability in probability_by_class.items()
    }
    # Two-decimal values can total 0.99 or 1.01. Put the rounding remainder on
    # the predicted class so the public probability distribution totals 1.00.
    remainder = round(1.0 - sum(rounded_probabilities.values()), 2)
    rounded_probabilities[risk_level] = round(
        rounded_probabilities[risk_level] + remainder, 2
    )
    confidence = rounded_probabilities[risk_level]

    return PredictResponse(
        riskLevel=risk_level,
        confidence=confidence,
        probabilities=RiskProbabilities(**rounded_probabilities),
        recommendation=build_recommendation(risk_level),
        modelVersion=model_version,
    )
