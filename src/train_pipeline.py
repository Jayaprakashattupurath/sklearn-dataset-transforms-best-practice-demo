"""
Train a robust scikit-learn pipeline for a churn-like use case.

Best-practice highlights:
- Keep preprocessing + model in ONE Pipeline (prevents train/serve skew)
- Use ColumnTransformer to handle numeric/categorical columns separately
- Fit preprocessors ONLY on training data
- Persist the entire pipeline as one artifact
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)
MODEL_PATH = ARTIFACT_DIR / "customer_churn_pipeline.joblib"


def build_demo_dataset() -> pd.DataFrame:
    # Small synthetic dataset; replace with your real CSV/DB extraction
    return pd.DataFrame(
        {
            "age": [24, 45, 37, 52, 29, 41, 34, 60, 48, 31],
            "monthly_spend": [20.5, 89.2, 45.0, 110.8, 33.3, 77.0, 55.4, 130.0, 95.2, 40.1],
            "tenure_months": [2, 24, 12, 48, 5, 36, 18, 60, 30, 8],
            "plan_type": ["basic", "pro", "basic", "enterprise", "basic", "pro", "pro", "enterprise", "pro", "basic"],
            "region": ["north", "south", "west", "east", "north", "west", "south", "east", "west", "north"],
            "churned": [1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        }
    )


def main() -> None:
    df = build_demo_dataset()
    X = df.drop(columns=["churned"])
    y = df["churned"]

    numeric_features = ["age", "monthly_spend", "tenure_months"]
    categorical_features = ["plan_type", "region"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=6,
        random_state=42,
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    print(f"Validation ROC-AUC: {auc:.3f}")
    joblib.dump(clf, MODEL_PATH)
    print(f"Saved pipeline to: {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    main()