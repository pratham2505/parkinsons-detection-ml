from pathlib import Path
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]

def main():
    # 1) Load trained model
    model_path = ROOT / "models" / "random_forest.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Trained model not found. Run: python -m src.train")

    model = joblib.load(model_path)

    # 2) Load dataset
    df = pd.read_csv(ROOT / "data" / "parkinsons.data")

    # 3) Load feature columns (same as training)
    with open(ROOT / "models" / "feature_columns.json", "r") as f:
        feature_cols = json.load(f)

    X = df[feature_cols]
    y = df["status"]

    # 4) Same train/test split as training (IMPORTANT)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5) Get prediction probabilities
    probs = model.predict_proba(X_test)[:, 1]

    # 6) Compute ROC curve + AUC
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)

    # 7) Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Parkinson's Detection")
    plt.legend(loc="lower right")

    # 8) Save figure
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    plt.savefig(reports_dir / "roc_curve.png", dpi=300)
    plt.close()

    print(f"ROC curve saved to {reports_dir / 'roc_curve.png'}")
    print(f"ROC-AUC: {auc:.4f}")

if __name__ == "__main__":
    main()
