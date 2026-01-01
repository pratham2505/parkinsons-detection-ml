from pathlib import Path
import json

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]

def main():
    # Load data
    df = pd.read_csv(ROOT / "data" / "parkinsons.data")

    y = df["status"]
    X = df.drop(columns=["status"])
    X = X.select_dtypes(exclude=["object"])

    # Save feature columns (so tuned model uses same order in predict)
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    with open(models_dir / "feature_columns.json", "w") as f:
        json.dump(list(X.columns), f)

    # Hold-out test set (never touched during tuning)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Base model
    rf = RandomForestClassifier(random_state=42)

    # Parameter grid (small + safe for your dataset)
    param_grid = {
        "n_estimators": [100, 300, 500],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": [None, "balanced"]
    }

    # We care about catching Parkinson’s -> optimize recall for class 1
    grid = GridSearchCV(
        rf,
        param_grid=param_grid,
        scoring="recall",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    print("\nBest parameters:", grid.best_params_)
    print("Best CV recall:", grid.best_score_)

    best_model = grid.best_estimator_

    # Evaluate on held-out test set
    preds = best_model.predict(X_test)
    probs = best_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()

    print("\n=== TUNED MODEL (TEST SET) ===")
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("\nClassification Report:\n", classification_report(y_test, preds))
    print("ROC-AUC:", auc)

    # Save tuned model
    out_path = models_dir / "random_forest_tuned.pkl"
    joblib.dump(best_model, out_path)
    print(f"\nTuned model saved to {out_path}")

if __name__ == "__main__":
    main()
