from pathlib import Path
import json

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]

def main():
    # Load model
    model_path = ROOT / "models" / "random_forest.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Baseline model not found. Run: python -m src.train")

    model = joblib.load(model_path)

    # Load data
    df = pd.read_csv(ROOT / "data" / "parkinsons.data")
    y = df["status"]

    # Load feature columns (same order as training)
    with open(ROOT / "models" / "feature_columns.json", "r") as f:
        feature_cols = json.load(f)

    X = df[feature_cols]

    # Same split as training
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Get probabilities for class 1 (Parkinson's)
    probs = model.predict_proba(X_test)[:, 1]

    print("\nThreshold tuning (positive class = 1 / Parkinson's)\n")
    print("thr   TN  FP  FN  TP   recall1  prec1")
    print("--------------------------------------")

    for thr in [i / 10 for i in range(1, 10)]:
        preds = (probs >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()

        rec1 = recall_score(y_test, preds, pos_label=1, zero_division=0)
        prec1 = precision_score(y_test, preds, pos_label=1, zero_division=0)

        print(f"{thr:.1f}  {tn:3d} {fp:3d} {fn:3d} {tp:3d}   {rec1:6.2f}   {prec1:6.2f}")

if __name__ == "__main__":
    main()
