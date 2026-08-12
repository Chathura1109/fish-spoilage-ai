"""Generate reproducible synthetic cold-chain snapshots for the prototype model.

The generated labels are engineering simulations for academic development only.
They are not expert food-safety labels and must be replaced with real inspection
outcomes before the model is used beyond decision-support demonstrations.
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "fish_spoilage_dataset.csv"
RANDOM_SEED = 20260812
BATCH_COUNT = 1_000
SNAPSHOTS_PER_BATCH = 4

SPECIES_SENSITIVITY = {
    "Tuna": 0.20,
    "Tilapia": 0.10,
    "Snapper": 0.25,
    "Sardine": 0.45,
    "Mackerel": 0.55,
}

FIELDNAMES = [
    "batch_id",
    "fish_species",
    "has_temperature_telemetry",
    "temperature_reading_count",
    "current_temperature",
    "average_temperature",
    "minimum_temperature",
    "maximum_temperature",
    "air_temperature",
    "humidity",
    "storage_duration_hours",
    "transport_duration_hours",
    "time_above_limit_minutes",
    "temperature_violation_count",
    "time_since_catch_hours",
    "spoilage_risk",
]


def clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def rounded(value: float) -> float:
    return round(value, 2)


def risk_label(score: float) -> str:
    if score >= 7.0:
        return "HIGH"
    if score >= 3.2:
        return "MEDIUM"
    return "LOW"


def generate_rows() -> list[dict[str, object]]:
    rng = random.Random(RANDOM_SEED)
    rows: list[dict[str, object]] = []

    for batch_number in range(1, BATCH_COUNT + 1):
        batch_id = f"SYNTH-BATCH-{batch_number:06d}"
        species = rng.choice(list(SPECIES_SENSITIVITY))
        final_age = rng.uniform(18.0, 168.0)
        cold_chain_baseline = rng.uniform(-0.5, 8.5)
        batch_humidity = rng.uniform(58.0, 96.0)
        batch_effect = rng.gauss(0.0, 0.45)

        for snapshot in range(1, SNAPSHOTS_PER_BATCH + 1):
            fraction = snapshot / SNAPSHOTS_PER_BATCH
            time_since_catch = final_age * fraction
            storage_duration = time_since_catch
            transport_duration = min(
                time_since_catch,
                max(0.0, time_since_catch * rng.uniform(0.12, 0.55)),
            )
            has_telemetry = rng.random() >= 0.07

            humidity = (
                clamp(batch_humidity + rng.gauss(0.0, 3.0), 35.0, 100.0)
                if rng.random() >= 0.03
                else None
            )
            air_temperature = (
                clamp(cold_chain_baseline + rng.gauss(0.8, 1.6), -5.0, 40.0)
                if rng.random() >= 0.03
                else None
            )

            if has_telemetry:
                reading_count = max(1, round(transport_duration * rng.uniform(1.5, 6.0)))
                warming = fraction * rng.uniform(0.0, 2.8)
                average_temperature = clamp(
                    cold_chain_baseline + warming + rng.gauss(0.0, 0.7), -5.0, 35.0
                )
                minimum_temperature = clamp(
                    average_temperature - rng.uniform(0.2, 2.8), -10.0, 35.0
                )
                maximum_temperature = clamp(
                    average_temperature + rng.uniform(0.3, 4.2), -4.5, 45.0
                )
                current_temperature = clamp(
                    average_temperature + rng.gauss(0.25, 1.0),
                    minimum_temperature,
                    maximum_temperature,
                )

                exposure_ratio = clamp(
                    0.08
                    + 0.12 * (average_temperature - 2.0)
                    + 0.08 * (maximum_temperature - 4.0),
                    0.0,
                    1.0,
                )
                time_above_limit = transport_duration * 60.0 * exposure_ratio
                violation_count = min(
                    reading_count,
                    max(0, round(reading_count * exposure_ratio + rng.gauss(0.0, 1.2))),
                )
            else:
                reading_count = 0
                current_temperature = None
                average_temperature = None
                minimum_temperature = None
                maximum_temperature = None
                time_above_limit = 0.0
                violation_count = 0

            temperature_score = 0.0
            if has_telemetry:
                temperature_score = (
                    max(0.0, float(average_temperature) - 2.0) * 0.34
                    + max(0.0, float(maximum_temperature) - 4.0) * 0.30
                    + (time_above_limit / 60.0) * 0.12
                    + (violation_count / max(1, reading_count)) * 1.8
                )
            else:
                # Missing product telemetry raises uncertainty but is not
                # treated as proof that a batch is unsafe.
                temperature_score = 1.4

            humidity_score = (
                max(0.0, humidity - 75.0) * 0.018 if humidity is not None else 0.25
            )
            duration_score = (time_since_catch / 24.0) * 0.48
            score = (
                temperature_score
                + humidity_score
                + duration_score
                + SPECIES_SENSITIVITY[species]
                + batch_effect
                + rng.gauss(0.0, 0.55)
            )

            rows.append(
                {
                    "batch_id": batch_id,
                    "fish_species": species,
                    "has_temperature_telemetry": has_telemetry,
                    "temperature_reading_count": reading_count,
                    "current_temperature": "" if current_temperature is None else rounded(current_temperature),
                    "average_temperature": "" if average_temperature is None else rounded(average_temperature),
                    "minimum_temperature": "" if minimum_temperature is None else rounded(minimum_temperature),
                    "maximum_temperature": "" if maximum_temperature is None else rounded(maximum_temperature),
                    "air_temperature": "" if air_temperature is None else rounded(air_temperature),
                    "humidity": "" if humidity is None else rounded(humidity),
                    "storage_duration_hours": rounded(storage_duration),
                    "transport_duration_hours": rounded(transport_duration),
                    "time_above_limit_minutes": rounded(
                        min(time_above_limit, rounded(transport_duration) * 60.0)
                    ),
                    "temperature_violation_count": violation_count,
                    "time_since_catch_hours": rounded(time_since_catch),
                    "spoilage_risk": risk_label(score),
                }
            )

    return rows


def main() -> None:
    rows = generate_rows()
    with DATASET_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    counts = {label: sum(row["spoilage_risk"] == label for row in rows) for label in ("LOW", "MEDIUM", "HIGH")}
    print(f"Wrote {len(rows)} rows from {BATCH_COUNT} batches to {DATASET_PATH.name}")
    print(f"Class counts: {counts}")


if __name__ == "__main__":
    main()
