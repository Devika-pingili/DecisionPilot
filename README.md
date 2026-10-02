# DecisionPilot

DecisionPilot is a personalized next-basket product recommendation system built from historical Instacart orders. It constructs point-in-time customer and product features, generates candidate products, and uses a saved `HistGradientBoostingClassifier` to rank them. The FastAPI backend serves recommendations and deterministic evidence; a Vue 3 dashboard presents the results and customer insights. An optional AI layer can explain verified evidence on request, but it does not select products or affect ranking.

## Problem Statement

Recommendations based only on general popularity can miss differences in customers' purchasing patterns. DecisionPilot uses historical context such as product purchase frequency, recency, reorder behavior, and department or aisle affinity to rank candidate products for a customer's next basket. These signals provide relevant context while keeping the prediction grounded in data available before the target order.

## Objectives

- Build a personalized next-basket recommendation system.
- Generate candidates from customer history and popularity signals.
- Construct features using only information available before each prediction target.
- Prevent target-order and future-order information from leaking into features.
- Train and load a supervised classification model for candidate scoring.
- Provide deterministic, evidence-based explanations.
- Serve recommendations through a FastAPI API and display them in an interactive dashboard.
- Keep AI explanations optional and prevent AI from changing recommendation rankings.

## Key Features

### Personalized Recommendations

Products are ranked for an individual customer using historical behavior and product/category signals. The model scores candidates supplied by the existing candidate generator; it does not score the entire catalog.

### Point-in-Time Prediction

For a target order numbered `N`, customer history is restricted to orders with `order_number < N`. The target order and later orders are excluded from feature construction. Cutoff-aware global product and category statistics are used where applicable.

### Candidate Generation

The deterministic candidate set can include:

- Products purchased previously by the customer, with frequent products prioritized.
- Products in the customer's recent orders.
- Products showing customer reorder behavior.
- Popular products in departments or aisles the customer has used.
- Global popularity candidates as a fallback source.

The generated union is bounded by the configured candidate/source limits.

### ML Ranking

The project uses scikit-learn's `HistGradientBoostingClassifier` as a binary classifier to estimate candidate scores. It is not a specialized learning-to-rank algorithm. Candidates are sorted by model score, with product ID as the deterministic tie-breaker. Scores are ranking values and are **not calibrated probabilities**.

### Explainable Recommendations

The recommendation service derives deterministic reason codes from verified point-in-time feature values:

| Reason code | Evidence represented |
| --- | --- |
| `CUSTOMER_FREQUENT` | The product appeared in multiple previous customer orders. |
| `CUSTOMER_RECENT` | The product appeared in the customer's recent orders. |
| `CUSTOMER_REORDER` | The customer's historical rows show reorder behavior for the product. |
| `DEPARTMENT_AFFINITY` | The customer has historical purchases in the product's department. |
| `AISLE_AFFINITY` | The customer has historical purchases in the product's aisle. |
| `GLOBAL_POPULARITY` | The product has prior purchase history across customers. |

The API returns these codes with deterministic explanation text. The dashboard renders the corresponding evidence badges.

### Customer Insights

The dashboard reuses the existing customer summary and recommendation responses to display purchase behavior, reason-code counts, highest and average displayed recommendation scores, recommendation count, and a top-recommendation highlight with its deterministic explanation.

### Optional AI Explanation

The user can request an explanation for the top recommendation. The backend verifies the product and reason codes using its own recommendation pipeline and sends only the product name and allowlisted evidence to the OpenAI provider. AI does not select, score, or rerank products and receives no customer/order identifiers or full purchase history. If AI is unavailable, the deterministic explanation remains available. Live provider generation depends on backend credentials, network access, and account quota; live generation is not confirmed for the current demo environment.

## System Architecture

```text
Instacart historical data
  |
  v
Data preparation and sparse history index
  |
  v
Point-in-time feature engineering
  |
  v
Candidate generation
  |
  v
ML feature dataset and saved classifier
  |
  v
Recommendation scoring and deterministic ranking
  |
  v
Verified reason codes and deterministic explanation
  |
  +----> Optional on-demand AI explanation
  |
  v
FastAPI backend
  |
  v
Vue 3 dashboard
```

- **Data preparation** validates source CSVs and builds prepared feature files.
- **History and features** use prior orders and cutoff-specific aggregates to construct customer, product, department, and aisle signals.
- **Candidate generation** combines customer-specific products with category/global popularity candidates.
- **ML dataset and model** use rolling prior-order targets, binary product-presence labels, and a saved classifier.
- **Recommendation service** scores and sorts candidates, then derives reason codes from the same point-in-time features.
- **FastAPI** exposes the customer summary, recommendations, and on-demand explanation endpoints.
- **Vue dashboard** presents recommendations, customer insights, deterministic evidence, and the optional AI explanation state.

## Data and Dataset

DecisionPilot uses the Instacart Market Basket Analysis data files:

- `orders.csv`
- `order_products__prior.csv`
- `order_products__train.csv`
- `products.csv`
- `aisles.csv`
- `departments.csv`

In the source data, `prior` contains observed order history, `train` contains labeled target-order products, and `test` represents an unlabeled future inference population. The project's rolling development and validation examples are constructed from prior orders; `order_products__train.csv` is not used to build prediction features.

The prediction unit is:

```text
(customer_id, target_order_id, target_order_number, candidate_product_id)
```

For a candidate product, `y = 1` if that product appears in the target order and `y = 0` otherwise. Target products are used to construct labels only; they are not passed into point-in-time feature construction.

The repository's current ML development artifact was built with a bounded configuration of the first 100 sorted eligible customers. It contains 250,583 candidate rows across 1,318 target groups, split into 189,868 training rows and 60,715 validation rows. These are local generated artifacts, not files committed to the repository.

Raw source data and processed outputs are expected locally under `data/raw/instacart/` and `data/processed/`. They are excluded from Git. A fresh clone does not include these data or model artifacts; the demo commands below assume the existing local artifacts are already available.

## Leakage Prevention

- For target order number `N`, history queries use the strict cutoff `order_number < N`.
- The target order's products and later orders are excluded from features.
- Target-order products are read separately to create labels; they are not inputs to the feature builder.
- Rolling development examples use `prior` orders. `order_products__train.csv` is not used for prediction features.
- Global product and category aggregates are queried at the target cutoff.
- Identifier columns and `product_name` are excluded from model predictors; IDs are used for grouping, joining, labels, and output association.
- Completed target-order outcomes such as target basket size, target reordered-item count, target reorder rate, and target `add_to_cart_order` are not model predictors. Historical, cutoff-aware customer/product reorder rates and basket-size statistics are separate features and are calculated from prior history.

Tests cover target/future-order exclusion, point-in-time cutoffs, and label/feature separation.

## Machine Learning

The bounded development dataset is built by generating candidates and point-in-time features for rolling prior-order targets, then labeling whether each candidate appears in that target basket. The split is chronological within each customer: the final `ceil(20%)` of eligible targets per customer are validation targets; earlier eligible targets are used for training. There is no random row shuffle for this split.

The model is a scikit-learn `HistGradientBoostingClassifier` configured in `backend/ml/model.py`. It is trained on numeric features after identifier and product-name columns are excluded. The saved `model.joblib` is loaded by the recommendation service; normal API startup does not retrain it. Inference sorts candidates by descending model score and ascending product ID to break ties deterministically.

Do not interpret model scores as calibrated probabilities.

## Evaluation

The following values are from the repository's bounded development/validation model artifacts. They describe the documented sample and configuration, not universal or production-wide performance.

| K | Precision@K | Recall@K | F1@K | MAP@K | NDCG@K |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0.385714 | 0.325402 | 0.309383 | 0.364485 | 0.462072 |
| 10 | 0.303247 | 0.489168 | 0.331639 | 0.347099 | 0.477559 |
| 20 | 0.218344 | 0.658340 | 0.298981 | 0.377631 | 0.536640 |

Candidate recall in the bounded ML validation configuration is **0.948052**. Candidate recall measures whether relevant target products were present in the generated candidate sets before ranking; it is not a ranking metric.

## Backend API

The FastAPI service exposes:

| Method and path | Purpose |
| --- | --- |
| `GET /` | Service information. |
| `GET /health` | Checks that recommendation resources are available. |
| `GET /customers/{customer_id}/summary` | Returns customer-level order and purchase summary data. |
| `GET /customers/{customer_id}/recommendations?top_k=5&order_number=...` | Returns ranked recommendations; `top_k` is 1–20 and `order_number` is an optional prior-order prediction cutoff. |
| `POST /customers/{customer_id}/recommendations/explain` | Requests an optional explanation for a product in the server-generated list. The JSON body accepts `product_id`, `top_k`, and optional `order_number`; evidence is derived by the backend. |
| `GET /diagnostics/ai` | Loopback-only diagnostic endpoint for optional AI configuration and provider status; not a primary user-facing API. |

## Frontend

The Vue 3 + Vite dashboard includes:

- Customer summary cards and customer ID input.
- Top 5, Top 10, and Top 20 controls plus an optional order cutoff.
- Recommendation cards with rank, product name, model score, and deterministic reason badges.
- Customer Insights with purchase behavior, reason breakdown, score summaries, and top recommendation details.
- A “How It Works” section.
- Deterministic explanations and a top-recommendation-only optional AI explanation action.
- Loading, error, and empty states, with responsive desktop and smaller-screen layouts.

## Technology Stack

| Area | Technologies |
| --- | --- |
| Backend | Python, FastAPI, Uvicorn, pandas, SQLite |
| ML | scikit-learn `HistGradientBoostingClassifier`, joblib, NumPy, SciPy |
| Frontend | Vue 3, Vite, JavaScript, CSS |
| Tests | pytest, FastAPI `TestClient`, HTTPX |
| Optional AI | Official OpenAI Python SDK |

## Project Structure

```text
DecisionPilot/
├── backend/
│   ├── ai/                 # Optional evidence explanation and diagnostics
│   ├── api/                # FastAPI app and schemas
│   ├── evaluation/         # Baselines and ranking metrics
│   ├── features/           # History, candidates, labels, point-in-time features
│   ├── ml/                 # Dataset builder, classifier, training/evaluation
│   ├── recommendation/    # Model loading, ranking, deterministic explanations
│   ├── inspect_data.py
│   └── prepare_data.py
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/     # Summary, recommendations, insights, states
│   │   ├── services/       # Backend API client
│   │   └── views/          # Dashboard
│   ├── package.json
│   ├── package-lock.json
│   └── vite.config.js
├── tests/
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

Raw datasets, the SQLite history index, saved model, and generated evaluation/training files are local artifacts and are excluded from Git. `.venv/`, `frontend/node_modules/`, and `frontend/dist/` are also ignored.

## Installation and Setup

Verified local runtime: Python 3.13.1. Install backend and test dependencies from PowerShell at the repository root:

```powershell
Set-Location D:\DecisionPilot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

The saved model was verified with `scikit-learn==1.8.0`, which is pinned in `requirements.txt`. Avoid changing that version without validating the saved model artifact. For this existing workspace, the prepared data, SQLite index, and saved model already exist locally; normal demo startup does not require data preparation, index rebuilding, or model training.

## Run the Application

Start the backend first in one PowerShell terminal:

```powershell
Set-Location D:\DecisionPilot
.\.venv\Scripts\python.exe -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
```

Then start the frontend in a second terminal:

```powershell
Set-Location D:\DecisionPilot\frontend
npm ci
npm run dev -- --host 0.0.0.0 --port 5173
```

Open the dashboard at [http://localhost:5173/](http://localhost:5173/). The frontend defaults to the API at `http://127.0.0.1:8000`; `VITE_API_BASE_URL` is an optional URL override, not a secret. Backend CORS allows the local frontend origin on port 5173.

The backend requires the already-prepared `data/processed/history.sqlite` and `data/processed/ml_dev/model.joblib`. These generated artifacts are not included in Git. Do not rerun data preparation or model training for normal startup when the local artifacts are present.

## Optional AI Configuration

The AI explanation layer is optional. To enable provider requests, configure these variables in the **backend process environment only** before starting Uvicorn:

- `DECISIONPILOT_AI_API_KEY`: OpenAI API credential.
- `DECISIONPILOT_AI_MODEL`: optional model name; the default is `gpt-4o-mini`.

Example for the current PowerShell session (replace the placeholder locally; never commit a real credential):

```powershell
$env:DECISIONPILOT_AI_API_KEY = '<your API key>'
$env:DECISIONPILOT_AI_MODEL = 'gpt-4o-mini'
```

The backend reads these values from its process environment; it does not load a `.env` file. Root `.gitignore` excludes `.env` and `.env.*`. Never put provider credentials in Vue code, frontend `VITE_` variables, or committed files.

AI is requested only when the user selects **Explain this recommendation**. The backend verifies the product and evidence first and sends only product name and verified reason evidence to the provider. AI does not change recommendations. Without a key, with provider errors/timeouts, or without usable provider quota, the deterministic explanation remains available. Live provider generation is not confirmed for the current demo environment; it depends on external provider configuration and account availability.

## Testing and Build

Run the Python test suite from the repository root:

```powershell
Set-Location D:\DecisionPilot
.\.venv\Scripts\python.exe -m pytest
```

Verified result: **108 passed, 0 failed, 0 skipped**, with one non-blocking Starlette/httpx deprecation warning.

Build the frontend production bundle:

```powershell
Set-Location D:\DecisionPilot\frontend
npm run build
```

The production build completed successfully in the verified environment.

## Demo Flow

1. Start the backend, then the frontend.
2. Open the dashboard and use Customer 1.
3. Review the customer summary and generate the default Top 5 list.
4. Change Top K to 10 and 20 and generate each list.
5. Inspect the top recommendation, score, and evidence badges.
6. Review Customer Insights and the “How It Works” section.
7. Show the deterministic explanation and click “Explain this recommendation.”
8. In the current provider-disabled demo environment, show the unavailable message while deterministic evidence and recommendations remain visible.

## Security and Data Handling

- Optional provider credentials are read by the backend from environment variables only.
- `.env` and `.env.*` are ignored; no API key belongs in source code or frontend variables.
- Raw Instacart data, SQLite history, model binaries, and generated training/evaluation outputs are not committed.
- `.venv/`, `frontend/node_modules/`, and `frontend/dist/` are ignored.
- The AI provider receives only the selected product name and verified evidence, not customer/order IDs or complete purchase history.

## Limitations

- Reported metrics are from a bounded development/validation configuration and should not be generalized to all customers or production traffic.
- Candidate generation bounds its sources and candidate list; it may not include every relevant product.
- Model scores are for ranking and are not calibrated probabilities.
- The project reflects the available historical Instacart data and its limitations.
- Live AI explanation depends on external provider availability, valid backend configuration, network access, and account quota; the core recommendation system does not depend on it.
- The current data/model workflow is a local project/demo setup, not a claim of production deployment or production-scale operation.

## Future Enhancements

Possible future work, not currently implemented, includes broader training and evaluation, ranking-specific models, additional temporal and personalization signals, online feedback and retraining, production storage/deployment, authentication, monitoring, and additional evaluation metrics.

## Project Context

DecisionPilot is an academic/portfolio project. No license file is present in the repository; no license is asserted here.
