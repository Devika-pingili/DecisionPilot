# 🚀 DecisionPilot — AI-Powered Next Basket Prediction

## 💡 Project Overview

**DecisionPilot** is a personalized next-basket prediction system that uses a customer's historical purchase behavior to rank products that are most likely to appear in their next order.

Instead of recommending only globally popular products, DecisionPilot combines customer-specific purchasing behavior with product, department, aisle, and global popularity signals.

### 🎯 Problem Statement

Given a customer's observed order history:

> **Which products are most likely to appear in the customer's next order?**

The system generates a set of candidate products, calculates historical features, uses a machine-learning model to score the candidates, and produces an ordered Top-K recommendation list with evidence-based explanations.

---

## 🛠️ Technologies Used

### Frontend

* Vue 3
* Vite
* JavaScript
* HTML
* CSS

### Backend

* Python
* FastAPI
* Uvicorn

### Machine Learning

* scikit-learn
* `HistGradientBoostingClassifier`
* joblib

### Data Processing

* Pandas
* NumPy
* CSV

### Database / Storage

* SQLite

### Development & Version Control

* VS Code
* Git
* GitHub

### Deployment

* Vercel — Frontend
* Render — Backend

### Dataset

* Instacart Market Basket Analysis Dataset
* Kaggle

### Optional AI Explanation

* OpenAI Python SDK

---

## ⚙️ Project Workflow

```text
Instacart Dataset
       ↓
Data Validation & Preparation
       ↓
Point-in-Time History
       ↓
Candidate Generation
       ↓
Feature Engineering
       ↓
Machine Learning Model
       ↓
Recommendation Scores
       ↓
Deterministic Ranking
       ↓
Evidence-Based Explanation
       ↓
FastAPI Backend
       ↓
Vue.js Frontend
       ↓
Personalized Recommendations
```

### 1. Data Preparation

The project validates the Instacart dataset for:

* Required columns
* Data types
* Value ranges
* Duplicate IDs
* Catalog relationships
* Missing joins
* Transaction key duplication
* Empty files

The processed data is then used for feature construction and recommendation generation.

### 2. Point-in-Time Feature Construction

For a target order with order number `N`, only historical orders satisfying:

```text
order_number < N
```

are used.

This prevents future information from leaking into the prediction features.

### 3. Candidate Generation

Candidate products are generated from:

* Historically purchased products
* Recent N-order products
* Frequently purchased products
* Customer-specific reorder behavior
* Department/aisle popularity
* Global popularity fallback

### 4. Feature Engineering

DecisionPilot creates features representing:

* Customer-product behavior
* Customer-level behavior
* Product/global popularity
* Department affinity
* Aisle affinity
* Historical reorder behavior
* Recency and frequency signals

All features are calculated using information available before the target order.

### 5. Machine Learning

The project uses:

**HistGradientBoostingClassifier**

The model produces a score for each candidate product.

These scores are used for ranking and are **not calibrated probabilities**.

### 6. Recommendation Ranking

Candidates are sorted by:

1. Model score — descending
2. Product ID — ascending for ties

The system supports:

* Top 5
* Top 10
* Top 20

### 7. Explainability

Recommendations include evidence-based reason codes such as:

* Frequent
* Recent
* Reordered
* Department Match
* Aisle Match
* Popular

---

# 🚀 How to Run the Project

## Prerequisites

Install:

* Python
* Node.js
* Git

Clone the repository:

```bash
git clone https://github.com/Devika-pingili/DecisionPilot.git
cd DecisionPilot
```

---

## Backend Setup

Create and activate a virtual environment.

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Start the FastAPI backend:

```powershell
uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
```

Backend will run at:

```text
http://127.0.0.1:8000
```

Health check:

```text
http://127.0.0.1:8000/health
```

---

# 🎨 Frontend Setup

Open a second terminal.

Move to the frontend directory:

```powershell
cd frontend
```

Install dependencies:

```powershell
npm install
```

Start the Vue development server:

```powershell
npm run dev
```

The frontend will normally be available at:

```text
http://localhost:5173
```

---

# 🔗 Frontend–Backend Connection

The frontend uses the environment variable:

```text
VITE_API_BASE_URL
```

For local development, the backend is:

```text
http://127.0.0.1:8000
```

For the deployed application, the frontend connects to the Render backend.

---

# 📊 Dataset Information

DecisionPilot uses the:

**Instacart Market Basket Analysis Dataset**

Source:

**Kaggle**

Dataset structure:

* `prior` — observed historical purchase history
* `train` — future labeled holdout
* `test` — future unlabeled inference population

The system uses historical information before the target order to construct prediction features.

`order_products__train.csv` is not used for feature construction.

---

# 🧠 Machine Learning Pipeline

```text
Historical Customer Orders
          ↓
Point-in-Time Cutoff
          ↓
Candidate Products
          ↓
Historical Features
          ↓
HistGradientBoostingClassifier
          ↓
Candidate Scores
          ↓
Deterministic Ranking
          ↓
Top-K Recommendations
```

---

# 📈 Evaluation

DecisionPilot evaluates recommendation quality using:

* Precision@K
* Recall@K
* F1@K
* MAP@K
* NDCG@K
* Candidate Recall

The evaluation metrics are calculated per customer and target order and then averaged.

The bounded development/validation configuration achieved a candidate recall of:

**0.948052**

---

# 🗄️ Scalable History Index

The complete prior transaction history contains:

**32,434,489 prior transactions**

The project uses a SQLite-backed history index for efficient cutoff-aware historical queries.

Full local history index:

```text
data/processed/history.sqlite
```

Approximate size:

**2.14 GB**

The full dataset is kept locally for development.

---

# 🌐 Deployment

DecisionPilot is deployed using:

### Frontend

**Vercel**

### Backend

**Render**

Architecture:

```text
GitHub
   ↓
Vercel
   ↓
Vue 3 Frontend
   ↓
Render
   ↓
FastAPI Backend
   ↓
DecisionPilot ML + Data
```

### Public Demo

Frontend:

https://decision-pilot-nine.vercel.app/

Backend:

https://decisionpilot.onrender.com

The public deployment uses a bounded demo dataset rather than uploading the complete 2.14 GB local history index.

---

# 🧪 Testing

The project includes automated tests covering:

* Data preparation
* Data validation
* History indexing
* Point-in-time behavior
* Candidate generation
* Labels
* Feature construction
* ML dataset construction
* Recommendation ranking
* Leakage prevention
* API behavior
* AI fallback behavior
* Demo-data behavior

Current test status:

**118 Python tests passed**

Frontend production build:

**Passed**

---

# 🤖 Optional AI Explanation

DecisionPilot contains an optional AI explanation layer.

The AI layer does **not**:

* Choose recommendations
* Rerank products
* Generate recommendation scores
* Replace the ML model

Instead, it verbalizes already verified recommendation evidence.

If the AI provider is unavailable, DecisionPilot falls back to its deterministic explanation.

---

# 🔐 Data & Security Considerations

* API keys remain backend-only.
* `.env` files are ignored.
* Raw datasets are not exposed publicly.
* Recommendation evidence is verified server-side.
* Client-provided reason codes are not trusted.
* Target-order information is excluded from feature construction.
* The complete local history database is not publicly deployed.

---

# 📁 Project Structure

```text
DecisionPilot/
│
├── backend/
│   ├── api/
│   ├── features/
│   ├── recommendation/
│   └── demo/
│
├── frontend/
│   └── src/
│       ├── components/
│       ├── services/
│       └── views/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── demo/
│
├── tests/
│
├── requirements.txt
├── requirements-dev.txt
├── README.md
└── .gitignore
```

---

# ⭐ Key Features

* Personalized next-basket prediction
* Point-in-time feature construction
* Leakage prevention
* Candidate generation
* Customer-specific behavioral signals
* Department and aisle affinity
* Global popularity fallback
* Machine-learning ranking
* Deterministic Top-K recommendations
* Evidence-based explanations
* Optional AI explanations
* SQLite-backed historical index
* FastAPI REST API
* Vue 3 interactive dashboard
* Automated testing
* Public bounded demo deployment

---

# 🎯 Conclusion

DecisionPilot combines historical purchase behavior, point-in-time feature engineering, candidate generation, machine learning, and full-stack web technologies to create personalized next-basket product rankings.

```text
Purchase History
      ↓
Point-in-Time Data
      ↓
Candidate Generation
      ↓
Feature Engineering
      ↓
ML Ranking
      ↓
Personalized Recommendations
      ↓
FastAPI
      ↓
Vue Dashboard
```

**DecisionPilot — using what a customer has bought before to rank what they may buy next.**
