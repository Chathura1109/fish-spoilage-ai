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
    """Telemetry aggregates accepted from the FishTrace Laravel backend."""

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
    currentProductTemperature: float | None = Field(default=None, ge=-5, le=40)
    averageProductTemperature: float | None = Field(default=None, ge=-5, le=40)
    minimumProductTemperature: float | None = Field(default=None, ge=-10, le=40)
    maximumProductTemperature: float | None = Field(default=None, ge=-5, le=50)
    airTemperature: float | None = Field(default=None, ge=-10, le=50)
    storageDurationHours: float = Field(ge=0, le=8760)
    transportDurationHours: float = Field(ge=0, le=720)
    humidity: float | None = Field(default=None, ge=0, le=100)
    timeAboveLimitMinutes: float = Field(ge=0, le=525_600)
    temperatureViolationCount: int = Field(ge=0, le=10_000)
    timeSinceCatchHours: float = Field(ge=0, le=8760)
    hasTemperatureTelemetry: bool
    temperatureReadingCount: int = Field(ge=0, le=1_000_000)

    @model_validator(mode="after")
    def validate_physical_relationships(self) -> "PredictRequest":
        """Reject fields that are valid alone but contradictory together."""
        product_temperatures = (
            self.currentProductTemperature,
            self.averageProductTemperature,
            self.minimumProductTemperature,
            self.maximumProductTemperature,
        )
        if self.hasTemperatureTelemetry:
            if self.temperatureReadingCount == 0 or any(
                value is None for value in product_temperatures
            ):
                raise ValueError(
                    "temperature statistics and a positive reading count are required "
                    "when hasTemperatureTelemetry is true"
                )
        elif (
            self.temperatureReadingCount != 0
            or any(value is not None for value in product_temperatures)
            or self.timeAboveLimitMinutes != 0
            or self.temperatureViolationCount != 0
        ):
            raise ValueError(
                "product-temperature statistics, exposure, and violations must be "
                "empty or zero when hasTemperatureTelemetry is false"
            )

        if self.temperatureViolationCount > self.temperatureReadingCount:
            raise ValueError(
                "temperatureViolationCount cannot exceed temperatureReadingCount"
            )

        if self.hasTemperatureTelemetry and not (
            self.minimumProductTemperature
            <= self.currentProductTemperature
            <= self.maximumProductTemperature
        ):
            raise ValueError(
                "currentProductTemperature must be between "
                "minimumProductTemperature and maximumProductTemperature"
            )
        if self.hasTemperatureTelemetry and not (
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
    speciesCategory: str
    speciesFallback: bool


def build_recommendation(
    risk_level: str,
    has_temperature_telemetry: bool,
    species_fallback: bool,
) -> str:
    """Convert the predicted class into a short operational recommendation."""
    if not has_temperature_telemetry:
        recommendation = (
            "Collect product-temperature telemetry and arrange a quality inspection."
        )
    elif risk_level == "HIGH":
        recommendation = (
            "Isolate the batch, maintain temperature below 4\u00b0C, and inspect immediately."
        )
    elif risk_level == "MEDIUM":
        recommendation = "Inspect the batch and maintain temperature below 4\u00b0C."
    else:
        recommendation = "Continue monitoring and maintain temperature below 4\u00b0C."

    if species_fallback:
        return (
            "Species-specific profile unavailable; prediction used the OTHER "
            f"category. {recommendation}"
        )
    return recommendation


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
    service_token = os.getenv("AI_SERVICE_TOKEN")
    if not service_token:
        raise HTTPException(status_code=503, detail="AI service is not configured")
    expected_authorization = f"Bearer {service_token}"
    if authorization is None or not secrets.compare_digest(
        authorization, expected_authorization
    ):
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Resolve exact names and supported aliases to a species seen in training.
    requested_species = request.fishSpecies.casefold()
    canonical_species = species_lookup.get(requested_species)
    if canonical_species is None:
        canonical_species = species_aliases.get(requested_species)
    species_fallback = False
    if canonical_species is None:
        canonical_species = species_lookup.get("other")
        species_fallback = True
    if canonical_species is None:
        raise HTTPException(status_code=503, detail="OTHER species model is unavailable")

    row = pd.DataFrame(
        [
            {
                "fish_species": canonical_species,
                "has_temperature_telemetry": int(request.hasTemperatureTelemetry),
                "temperature_reading_count": request.temperatureReadingCount,
                "current_temperature": request.currentProductTemperature,
                "average_temperature": request.averageProductTemperature,
                "minimum_temperature": request.minimumProductTemperature,
                "maximum_temperature": request.maximumProductTemperature,
                "air_temperature": request.airTemperature,
                "storage_duration_hours": request.storageDurationHours,
                "humidity": request.humidity,
                "transport_duration_hours": request.transportDurationHours,
                "time_above_limit_minutes": request.timeAboveLimitMinutes,
                "temperature_violation_count": request.temperatureViolationCount,
                "time_since_catch_hours": request.timeSinceCatchHours,
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
        recommendation=build_recommendation(
            risk_level,
            request.hasTemperatureTelemetry,
            species_fallback,
        ),
        modelVersion=model_version,
        speciesCategory=canonical_species,
        speciesFallback=species_fallback,
    )
