# DecisionPilot

DecisionPilot predicts products a customer may need in a next basket from historical Instacart orders. Its ranking is based on a saved supervised model and point-in-time purchase evidence. An optional AI service can paraphrase verified evidence for the top recommendation; it never selects or ranks products.

## System Flow

```text
Historical Instacart data
  -> point-in-time history and features
  -> deterministic candidate generation
  -> saved ML ranking model
  -> ranked recommendations with deterministic reason codes
  -> optional on-demand AI explanation
```

For a prediction at order number `N`, customer-specific history is restricted to prior orders with `order_number < N`. The target order and all later orders are excluded from features. Candidate generation builds the allowed product set from customer and popularity signals. The existing `HistGradientBoostingClassifier` scores those candidates; sorting and deterministic tie-breaking produce the returned ranking. The saved model is loaded at API startup and is not retrained during normal application startup or recommendation requests.

The recommendation API attaches deterministic explanation text and allowlisted reason codes to each result. The Vue dashboard displays these explanations, customer summary metrics, a reason-code breakdown, score summaries, and a top recommendation. The AI explanation is a separate, explicit action for that top recommendation only. When AI is unconfigured or unavailable, deterministic explanations and recommendations continue to work normally.

## Repository Layout

- `backend/features/`: historical data access, candidate generation, labels, and point-in-time features.
- `backend/ml/`: bounded training data preparation, model training, and evaluation.
- `backend/recommendation/`: saved-model loading and ranked recommendation generation.
- `backend/api/`: FastAPI endpoints and response schemas.
- `backend/ai/`: optional provider-backed explanation of server-verified reason codes.
- `frontend/src/`: Vue dashboard, customer insights, API client, and presentation.
- `tests/`: backend unit and API tests using small fixtures.
- `data/raw/` and `data/processed/`: local source data and generated artifacts; both are excluded from Git.

## Requirements

- Python 3.13 (the verified local runtime) and Node.js/npm compatible with the Vite project.
- Raw Instacart source files in `data/raw/instacart/` when rebuilding local data/model artifacts.
- A configured OpenAI API key only if an actual AI explanation is desired. AI is optional.

Install runtime and test dependencies from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

`scikit-learn==1.8.0` is pinned because it is the verified compatible runtime for the saved `model.joblib` artifact. Do not upgrade it without validating that artifact. `numpy`, `scipy`, and `joblib` are used by the model stack; `joblib` is declared directly because the project imports it directly.

## Local Data and Model Artifacts

The six source files expected under `data/raw/instacart/` are `orders.csv`, `order_products__prior.csv`, `order_products__train.csv`, `products.csv`, `aisles.csv`, and `departments.csv`.

To recreate the prepared data, bounded development dataset, and saved model from the local raw data, run from the repository root:

```powershell
.\.venv\Scripts\python.exe backend\prepare_data.py
.\.venv\Scripts\python.exe -c "from pathlib import Path; from backend.features.history import HistoryStore; from backend.ml.dataset import build_development_dataset; raw=Path('data/raw/instacart'); index=Path('data/processed/history.sqlite'); history=HistoryStore.from_csv(raw, index_path=index); build_development_dataset(history, customer_limit=100, output_dir=Path('data/processed/ml_dev')); history.close()"
.\.venv\Scripts\python.exe backend\ml\train.py
```

These commands build local data and train a model. They are a reproduction workflow, not part of demo startup. For an existing workspace, keep and use its generated `data/processed/history.sqlite` and `data/processed/ml_dev/model.joblib`; neither needs to be rebuilt to run the app. Raw data, the SQLite index, training outputs, model files, and evaluation outputs are intentionally not committed.

## Run the Demo

Start the backend in one PowerShell terminal from the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
```

Install and start the frontend in a second terminal:

```powershell
Set-Location frontend
npm ci
npm run dev -- --host 0.0.0.0 --port 5173
```

Open `http://localhost:5173/`. The API defaults to `http://127.0.0.1:8000`; set the frontend build-time variable `VITE_API_BASE_URL` only when the API is hosted elsewhere. This is a URL, not a credential.

The backend requires the local SQLite index and saved model artifact. If either is unavailable, the API reports that its recommendation resources are not ready; the frontend does not contain or load model/data artifacts.

## Optional AI Explanation

The provider is OpenAI using the official Python SDK. Configure these variables in the backend process environment only:

- `DECISIONPILOT_AI_API_KEY`: optional provider credential.
- `DECISIONPILOT_AI_MODEL`: optional model name; defaults to `gpt-4o-mini`.

For a current PowerShell session, set them before starting the backend:

```powershell
$env:DECISIONPILOT_AI_API_KEY = '<your OpenAI API key>'
$env:DECISIONPILOT_AI_MODEL = 'gpt-4o-mini'
```

Do not put the key in Vue code, `VITE_` variables, or committed files. `.env` files are ignored by Git, but this application reads the key from the backend process environment and does not require a `.env` file.

AI is called only after the user selects **Explain this recommendation**. The backend regenerates and validates the selected product using the existing recommendation pipeline, then sends the product name and allowlisted factual reason evidence. It does not send customer IDs, order IDs, or purchase history. Provider output is plain explanatory text; recommendation ranking and deterministic reason codes remain authoritative. With no key, a provider error, or a timeout, the API returns a clean unavailable response and the application continues to show recommendations and deterministic explanations.

## API

- `GET /health`
- `GET /customers/{customer_id}/summary`
- `GET /customers/{customer_id}/recommendations?top_k=5&order_number=10`
- `POST /customers/{customer_id}/recommendations/explain`

The explanation request accepts `product_id`, `top_k`, and optional `order_number`. It does not accept authoritative reason codes; the backend derives evidence from its own ranked recommendation result. Existing recommendation response fields are unchanged.

## Tests and Build

Run the full Python test suite from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Build the frontend production bundle:

```powershell
Set-Location frontend
npm run build
```
