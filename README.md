# Fish Spoilage Risk Prediction API

AI component of the **IoT-Based Fish Traceability and Cold-Chain Monitoring System** university project.

This repository trains a Random Forest classifier from prototype fish cold-chain data and serves its predictions through FastAPI. The service predicts one of three risk levels:

- `LOW`
- `MEDIUM`
- `HIGH`

> **Important:** The current dataset is synthetic. This model is a university prototype and decision-support tool, not a food-safety certification system. An authorised quality inspector must make the final decision about a fish batch.

## Current model results

The saved `spoilage-v3` model was evaluated on a group-held-out test set.

| Metric | Result |
|---|---:|
| Rows originally loaded | 4,000 |
| Contradictory rows rejected | 0 |
| Rows used | 4,000 |
| Unique synthetic batches | 1,000 |
| Training rows / batches | 3,200 / 800 |
| Test rows / batches | 800 / 200 |
| Train/test batch overlap | 0 |
| Test accuracy | 88.12% |
| HIGH-risk precision | 91.61% |
| HIGH-risk recall | 91.92% |
| HIGH-risk F1-score | 91.76% |

The complete evaluation is stored in [`model_metrics.json`](model_metrics.json), and the visual result is stored in [`confusion_matrix.png`](confusion_matrix.png).

## Main features

- Validates and cleans the training CSV before model fitting.
- Rejects contradictory temperature and duration records.
- Prevents batch leakage between training and testing.
- One-hot encodes fish species.
- Imputes missing feature values.
- Gives additional training weight to the `HIGH` class.
- Calibrates the model probabilities.
- Returns probabilities for all three risk levels.
- Provides strict FastAPI request validation.
- Includes interactive Swagger documentation.
- Includes automated API and artifact regression tests.

## How the system works

```text
fish_spoilage_dataset.csv
          |
          v
Dataset validation and cleaning
          |
          v
Group-aware train/test split
          |
          v
One-hot encoding + missing-value imputation
          |
          v
Random Forest + probability calibration
          |
          v
spoilage_model.joblib
          |
          v
FastAPI POST /predict
          |
          v
Risk level + confidence + probabilities + recommendation
```

## Project files

| File | Purpose |
|---|---|
| `generate_dataset.py` | Reproducibly generates grouped synthetic prototype snapshots |
| `fish_spoilage_dataset.csv` | Synthetic model-training data |
| `train_model.py` | Validates data, trains the model, evaluates it, and saves artifacts |
| `predict_api.py` | Loads the artifact and exposes the FastAPI endpoints |
| `spoilage_model.joblib` | Saved preprocessing pipeline, calibrated model, and metadata |
| `model_metrics.json` | Machine-readable data audit and evaluation results |
| `confusion_matrix.png` | Visual evaluation of the held-out test predictions |
| `test_predict_api.py` | Automated API, validation, and artifact tests |
| `requirements.txt` | Python dependencies |

## Requirements

- Python 3.12 recommended
- Windows, Linux, or macOS
- `pip` or `uv`

## Installation

### Option 1: Standard Python and pip

Create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On Linux or macOS, activate it with:

```bash
source .venv/bin/activate
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

### Option 2: uv

```powershell
uv venv --python 3.12
uv pip install -r requirements.txt
```

You can also run commands without manually activating an environment:

```powershell
uv run --python 3.12 --with-requirements requirements.txt python train_model.py
```

## Training the model

Regenerate the deterministic synthetic prototype dataset when its rules change:

```powershell
python generate_dataset.py
```

Run:

```powershell
python train_model.py
```

Or with `uv`:

```powershell
uv run --python 3.12 --with-requirements requirements.txt python train_model.py
```

Training performs the following operations:

1. Loads `fish_spoilage_dataset.csv`.
2. Checks that every required column exists.
3. Normalises species names and risk labels.
4. Converts numeric columns safely.
5. Rejects impossible values and relationships.
6. Removes exact duplicate samples.
7. Requires and validates stable batch IDs (with a fallback only for legacy CSVs).
8. Creates a group-aware training and test split.
9. Builds group-aware probability-calibration folds.
10. Fits preprocessing and the Random Forest together.
11. Evaluates the untouched test fold.
12. Saves the model, metrics, and confusion matrix.

The command updates these files:

- `spoilage_model.joblib`
- `model_metrics.json`
- `confusion_matrix.png`

## Training dataset schema

The current CSV requires these columns:

| Column | Type | Meaning |
|---|---|---|
| `batch_id` | Text | Stable batch group used to prevent train/test leakage |
| `fish_species` | Text | Fish category known to the model |
| `has_temperature_telemetry` | Boolean | Whether product-temperature statistics are available |
| `temperature_reading_count` | Integer | Number of product-temperature readings |
| `current_temperature` | Number | Current product temperature in degrees Celsius |
| `average_temperature` | Number | Average product temperature in degrees Celsius |
| `minimum_temperature` | Number | Lowest product temperature in degrees Celsius |
| `maximum_temperature` | Number | Highest product temperature in degrees Celsius |
| `air_temperature` | Number | Latest container air temperature in degrees Celsius |
| `storage_duration_hours` | Number | Total storage duration in hours |
| `humidity` | Number | Relative humidity percentage |
| `transport_duration_hours` | Number | Transport duration in hours |
| `time_above_limit_minutes` | Number | Total measured time above the configured temperature limit |
| `temperature_violation_count` | Integer | Number of recorded temperature violations |
| `time_since_catch_hours` | Number | Time elapsed since the earliest linked catch |
| `spoilage_risk` | Text | Target label: `LOW`, `MEDIUM`, or `HIGH` |

Multiple snapshots from the same physical fish batch must have the same batch ID. The splitter keeps the complete batch on only one side of the train/test boundary.

## Dataset validation

The training script rejects records when:

- A target label is missing or is not `LOW`, `MEDIUM`, or `HIGH`.
- The fish species is missing.
- A numeric value falls outside the broad prototype engineering limits.
- A reading or violation count is not a whole number.
- Current product temperature is greater than maximum product temperature.
- Average product temperature is greater than maximum product temperature.
- Minimum product temperature is greater than current or average temperature.
- Transport duration is greater than storage duration.
- Storage duration is greater than time since catch.
- Time above the limit exceeds the total storage duration.
- Telemetry availability contradicts its statistics or reading count.
- Violation count exceeds reading count.

Missing feature values are not rejected. They are imputed inside the model pipeline using the most frequent species or numeric training median.

## Running FastAPI

Set a strong service token, then start the server:

```powershell
$env:AI_SERVICE_TOKEN = "replace-with-a-long-random-secret"
uvicorn predict_api:app --reload --host 127.0.0.1 --port 8000
```

Or run it with `uv`:

```powershell
uv run --python 3.12 --with-requirements requirements.txt uvicorn predict_api:app --reload --port 8000
```

Available URLs:

| URL | Purpose |
|---|---|
| `http://127.0.0.1:8000/` | Health check |
| `http://127.0.0.1:8000/predict` | Prediction endpoint |
| `http://127.0.0.1:8000/docs` | Swagger UI |
| `http://127.0.0.1:8000/redoc` | ReDoc documentation |
| `http://127.0.0.1:8000/openapi.json` | OpenAPI schema |

## Connecting to the hosted Vercel API

The ML API is already hosted on Vercel. Other team members do **not** need a
Vercel account, the Vercel CLI, this Python environment, or a separate
deployment. They only need connection details from the team member who manages
the hosted service:

```text
VERCEL_BASE_URL=https://fish-spoilage-ai.vercel.app
AI_SERVICE_TOKEN=the-shared-backend-service-token
```

Share the real token through a private channel. Do not commit it to Git, include
it in screenshots, send it to a mobile application, or store it in IoT firmware.

### Backend team-member setup

1. Obtain the production `VERCEL_BASE_URL` and `AI_SERVICE_TOKEN` from the Vercel
   project owner.
2. Open the backend project's local `.env` file.
3. Add the hosted base URL and token using the environment-variable names
   expected by that backend. For a Laravel backend, a recommended configuration
   is:

   ```dotenv
   AI_SERVICE_URL=https://fish-spoilage-ai.vercel.app
   AI_SERVICE_TOKEN=replace-with-the-private-shared-token
   ```

4. Ensure the backend appends `/predict` only once. The final prediction URL
   must look like:

   ```text
   https://fish-spoilage-ai.vercel.app/predict
   ```

5. If Laravel configuration was already cached, refresh it and restart the
   backend process:

   ```powershell
   php artisan config:clear
   ```

6. Confirm that the backend sends these HTTP headers:

   ```http
   Content-Type: application/json
   Authorization: Bearer the-same-value-as-AI_SERVICE_TOKEN
   ```

7. Test the hosted health endpoint before testing a prediction:

   ```powershell
   Invoke-RestMethod -Method Get `
     -Uri "https://fish-spoilage-ai.vercel.app/"
   ```

   A successful response contains `"status": "AI service running"`.

8. Send the complete JSON body shown in the **Prediction endpoint** section to
   `/predict`. A successful request returns HTTP `200` and the risk prediction.

The variable names in the example are recommendations. If the separate backend
repository uses different configuration names, use the names defined in that
backend's code rather than creating unused variables.

### IoT team-member setup

The IoT device should not call the Vercel `/predict` endpoint directly. It
normally has only raw sensor readings, while the ML API requires aggregated
values including current, average, minimum, and maximum product temperature,
reading count, violation count, and duration values.

The intended connection is:

```text
IoT device
    -> sends raw sensor readings to Firebase or the Laravel backend
Laravel backend
    -> validates and stores the readings
    -> calculates all required aggregate values
    -> calls the hosted Vercel /predict endpoint
Vercel ML API
    -> returns the prediction to the Laravel backend
```

The IoT team member therefore needs the Firebase or Laravel telemetry endpoint
and its device credentials—not `AI_SERVICE_TOKEN`. Test the connection in this
order:

1. Confirm the device has Wi-Fi access.
2. Confirm raw readings arrive in Firebase or the Laravel backend.
3. Confirm the backend can retrieve and aggregate those readings.
4. Confirm the backend calls the hosted `/predict` URL.
5. Confirm the prediction is stored or returned to the application.

### Hosted API status codes

| Status | Meaning in this project |
|---:|---|
| `200` | The hosted service accepted the request and returned a prediction |
| `401` | The backend omitted the bearer token or used the wrong token |
| `404` | The hosted domain or `/predict` path is incorrect |
| `405` | `/predict` was called with a method other than `POST` |
| `422` | The backend JSON is incomplete, incorrectly named, or physically inconsistent |
| `500` | The hosted function raised an unexpected runtime error |
| `503` | The hosted service token or required model artifact is unavailable |

For `422`, print or record the complete response body. FastAPI's `detail` list
identifies the exact invalid field. A body containing only raw fields such as
`temperature`, `humidity`, or `deviceId` cannot be sent directly to `/predict`.
The backend must construct the complete request documented below.

## Health-check endpoint

Request:

```http
GET /
```

Example response:

```json
{
  "status": "AI service running",
  "modelVersion": "spoilage-v3",
  "datasetType": "synthetic prototype",
  "decisionSupportOnly": true
}
```

## Prediction endpoint

Request:

```http
POST /predict
Content-Type: application/json
Authorization: Bearer replace-with-a-long-random-secret
```

The prediction endpoint returns HTTP `401` when the bearer token is missing or
does not match the server's `AI_SERVICE_TOKEN`. The health-check endpoint stays
public.

Example body:

```json
{
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
  "hasTemperatureTelemetry": true,
  "temperatureReadingCount": 48
}
```

Example response from the current model:

```json
{
  "riskLevel": "MEDIUM",
  "confidence": 0.89,
  "probabilities": {
    "LOW": 0.06,
    "MEDIUM": 0.89,
    "HIGH": 0.05
  },
  "recommendation": "Inspect the batch and maintain temperature below 4 degrees C.",
  "modelVersion": "spoilage-v3",
  "speciesCategory": "Tuna",
  "speciesFallback": false
}
```

Probabilities are rounded to two decimal places and adjusted for rounding so they total `1.00`. `confidence` is the probability belonging to `riskLevel`.

## Prediction request fields

| Field | Allowed range | Used by model? | Description |
|---|---:|:---:|---|
| `fishSpecies` | 1-100 characters | Yes | Fish species name |
| `currentProductTemperature` | -5 to 40°C | Yes | Latest product-sensor reading |
| `averageProductTemperature` | -5 to 40°C | Yes | Average product temperature |
| `minimumProductTemperature` | -10 to 40°C or `null` | Yes | Minimum product temperature |
| `maximumProductTemperature` | -5 to 50°C | Yes | Maximum product temperature |
| `airTemperature` | -10 to 50°C or `null` | Yes | Box air temperature |
| `humidity` | 0 to 100% or `null` | Yes | Relative humidity |
| `storageDurationHours` | 0 to 8,760 | Yes | Total storage duration |
| `transportDurationHours` | 0 to 720 | Yes | Transport duration |
| `timeAboveLimitMinutes` | 0 to 525,600 | Yes | Total time over the temperature limit |
| `temperatureViolationCount` | 0 to 10,000 | Yes | Count of temperature violations |
| `timeSinceCatchHours` | 0 to 8,760 | Yes | Time elapsed since catch |
| `hasTemperatureTelemetry` | `true` or `false` | Yes | Whether temperature telemetry is available |
| `temperatureReadingCount` | 0 to 1,000,000 | Yes | Number of available temperature readings |

Product-temperature summary fields may be `null` only when `hasTemperatureTelemetry` is `false`. Air temperature and humidity may be `null`; the saved preprocessing pipeline imputes those missing values.

## Cross-field request validation

FastAPI rejects a request with HTTP `422` when:

- Current product temperature is outside the minimum/maximum interval.
- Average product temperature is outside the minimum/maximum interval.
- Transport duration is greater than storage duration.
- Storage duration is greater than time since catch.
- Time above limit is longer than the complete storage duration.
- Telemetry availability, reading count, and temperature statistics contradict one another.
- Temperature violations exceed the number of readings.
- A numeric field is infinite or `NaN`.
- An unexpected extra field is supplied.

Species matching is case-insensitive. `Yellowfin Tuna` maps to the broader `Tuna` category. Any other unsupported species maps to the trained heterogeneous `Other` category and returns `speciesFallback=true`; its recommendation clearly requires cautious interpretation and quality inspection.

## Calling the API with curl

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AI_SERVICE_TOKEN" \
  -d '{
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
    "hasTemperatureTelemetry": true,
    "temperatureReadingCount": 48
  }'
```

## Calling the API from Python

```python
import requests

payload = {
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

response = requests.post(
    "http://127.0.0.1:8000/predict",
    headers={"Authorization": "Bearer replace-with-a-long-random-secret"},
    json=payload,
    timeout=10,
)
response.raise_for_status()
print(response.json())
```

## FishTrace Laravel integration

The FishTrace Laravel backend should:

1. Calculate summary values from stored IoT readings.
2. Send the JSON request to `POST /predict` with the shared bearer token in the
   `Authorization` header.
3. Store the full response in the `ai_predictions` table.
4. Show the risk as decision support.
5. Generate an alert when `riskLevel` is `HIGH`.
6. Never use this response as automatic food-safety certification.

In production, FishTrace calls the deployed HTTPS base URL and appends `/predict`. Firebase is only the real-time telemetry buffer; Laravel imports validated readings into MySQL before aggregating these model features.

## Running tests

Run only the project test file:

```powershell
pytest -q test_predict_api.py
```

Or with `uv`:

```powershell
uv run --python 3.12 --with-requirements requirements-dev.txt pytest -q test_predict_api.py
```

The tests cover:

- Health endpoint metadata.
- Valid prediction response structure.
- Case-insensitive species and Yellowfin Tuna aliasing.
- Probability ranges and sum.
- Confidence matching the selected class probability.
- Negative durations.
- Invalid humidity.
- Contradictory temperature summaries.
- Contradictory elapsed times.
- Unknown-species OTHER fallback and disclosure metadata.
- Unexpected fields.
- Missing product-temperature telemetry.
- Contradictory telemetry availability metadata.
- Batch leakage and HIGH-risk recall in the saved metrics.

## Model design

The saved artifact contains:

- Most-frequent imputation for missing species.
- One-hot encoding for fish species.
- Median imputation for numeric inputs.
- A 300-tree Random Forest with maximum depth 10.
- Minimum leaf size 3.
- Explicit class weights: LOW `1.0`, MEDIUM `1.0`, HIGH `2.0`.
- Sigmoid probability calibration using group-aware folds.
- Model version, training timestamp, feature list, species list, metrics, and safety metadata.

The model is deterministic because training and splitting use `random_state=42`.

## Confusion matrix

The current held-out confusion matrix is:

| True / Predicted | LOW | MEDIUM | HIGH |
|---|---:|---:|---:|
| LOW | 217 | 27 | 0 |
| MEDIUM | 19 | 215 | 25 |
| HIGH | 0 | 24 | 273 |

The model missed 24 HIGH-risk samples by predicting MEDIUM, but none were predicted LOW.

## Limitations

- The dataset and labels are AI-generated synthetic data.
- Synthetic performance does not prove performance on real fish batches.
- The current file contains four synthetic time snapshots for each generated batch.
- Yellowfin Tuna is mapped to Tuna rather than learned independently.
- Unsupported species use a heterogeneous synthetic OTHER category; this is not a species-specific biological profile.
- The temperature limits in the code are engineering validation ranges, not official safety thresholds.
- Prediction probabilities are calibrated only against synthetic data.
- The model does not use odour, texture, eye condition, gill colour, laboratory testing, or expert inspection results.

## Recommended next steps

1. Collect real sensor readings and expert-labelled inspection outcomes.
2. Preserve the real `batch_id` with every training observation.
3. Define species-specific handling with a fisheries or food-science supervisor.
4. Retrain and evaluate on batches from different trips, dates, boats, and routes.
5. Compare the Random Forest against a simple rule-based baseline.
6. Recheck probability calibration on independent real data.
7. Monitor model drift after deployment.

## Troubleshooting

### `spoilage_model.joblib not found`

Run the training script before starting FastAPI:

```powershell
python train_model.py
```

### Unknown fish species

The trained categories are Tuna, Tilapia, Snapper, Sardine, Mackerel, and Other. Yellowfin Tuna uses an explicit Tuna alias. Any other name (for example, Seer Fish) uses Other and returns `speciesFallback=true`; inspect that response as a generic telemetry-based estimate rather than species-specific evidence.

### HTTP `422 Unprocessable Entity`

Read the `detail` array in the response. It identifies the missing, out-of-range, contradictory, or unexpected field.

### Port 8000 is already in use

Start the server on another port:

```powershell
uvicorn predict_api:app --reload --port 8001
```

### Model artifact is outdated

If the API reports missing bundle keys, regenerate it:

```powershell
python train_model.py
```

## Academic use

This code is intended for the AI component of a university prototype. Clearly disclose the synthetic nature of the dataset, report both successful and failed predictions, and avoid presenting its thresholds or probabilities as approved food-safety standards.
