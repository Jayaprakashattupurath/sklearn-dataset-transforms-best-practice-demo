# scikit-learn Dataset Transformation Best Practice Demo

A small, practical template for a **real-time churn scoring use case** using:

- **Python + scikit-learn** for model training and preprocessing
- **FastAPI** for serving predictions
- **React (Vite)** frontend for a lightweight user input flow

Reference docs:
- scikit-learn: https://scikit-learn.org/stable/
- Data transformations: https://scikit-learn.org/stable/data_transforms.html

## Why this setup is a best practice

This project follows key transformation best practices from scikit-learn:

1. **Single Pipeline for preprocessing + model**  
   Prevents mismatch between training-time and inference-time transformations.

2. **ColumnTransformer for mixed data types**  
   Numeric features use imputation + scaling, categorical features use imputation + one-hot encoding.

3. **Fit transforms only on training split**  
   Avoids data leakage.

4. **Persist one artifact (`joblib`)**  
   Simplifies deployment and ensures exact same preprocessing in production.

5. **`OneHotEncoder(handle_unknown="ignore")`**  
   Handles unseen categories safely in real-time inference.

---

## Project structure

- `src/train_pipeline.py` → train and save pipeline artifact
- `src/api.py` → REST API with `/predict`
- `frontend/src/App.jsx` → minimal React form calling API
- `requirements.txt` → backend dependencies

---

## Quick start

## 1) Backend (Python)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python src/train_pipeline.py
uvicorn src.api:app --reload --port 8000
```

Test quickly:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age":38,"monthly_spend":59.9,"tenure_months":15,"plan_type":"pro","region":"west"}'
```

## 2) Frontend (React + Node)

```bash
cd frontend
npm install
npm run dev
```

Open the URL printed by Vite (usually `http://localhost:5173`).

---

## Real-time scenario guidance

Imagine this as part of a CRM workflow:
- Support or retention team opens the app
- Inputs customer profile
- API returns churn risk
- High-risk users are routed to retention campaigns

### How to evolve this for production

- Replace demo dataset with your warehouse extraction.
- Add proper train/validation/test or cross-validation.
- Track metrics over time (AUC, precision-recall by threshold).
- Version model artifacts (e.g., with MLflow or a model registry).
- Add request validation, auth, logging, and monitoring in API.
- Deploy frontend and API behind HTTPS and CORS policy.

---

## Notes

This is intentionally lightweight code, but organized to reflect a realistic production direction while preserving scikit-learn transformation best practices.