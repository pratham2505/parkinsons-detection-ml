from pathlib import Path
import pandas as pd # loads datasets file into a table-like structure
import joblib # saves and loads trained model files
from sklearn.model_selection import train_test_split # splits datasets into training and testing sets
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # different ways to measure how good my model is
from sklearn.ensemble import RandomForestClassifier # the model that we are training
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import json
import argparse

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-smote", action="store_true", help="Also train + save a SMOTE model")
    args = parser.parse_args()

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    df = pd.read_csv(ROOT / "data" / "parkinsons.data") # first we read datafile into a pandas dataframe

    y = df["status"] # y is the answer which we want the model to predict (0 = no parkinsons, 1 = parkinsons)
    X = df.drop(columns=["status"]) # X is the input information that we use to guess the answer (all the voice features)

    # drop any non-numeric columns (like the 'name' ID)
    X = X.select_dtypes(exclude=["object"])

    with open(models_dir / "feature_columns.json", "w") as f:
        json.dump(list(X.columns), f)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y # 0.2 is 20% test 80% train, same split every time so the results are repeatable
    ) # stratify=y keeps the same % of healthy/parkinson in both sets (important for fairness)

    # 1) Baseline model (no SMOTE)
    baseline = RandomForestClassifier(random_state=42)
    baseline.fit(X_train, y_train)

    baseline_preds = baseline.predict(X_test)
    baseline_probs = baseline.predict_proba(X_test)[:, 1]
    baseline_auc = roc_auc_score(y_test, baseline_probs)

    print("\n=== BASELINE (no SMOTE) ===")
    print("Accuracy:", accuracy_score(y_test, baseline_preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, baseline_preds))
    print("Classification Report:\n", classification_report(y_test, baseline_preds))
    print("ROC-AUC:", baseline_auc)

    baseline_path = models_dir / "random_forest.pkl"
    joblib.dump(baseline, baseline_path)
    print(f"\nBaseline model saved to {baseline_path}")

    # 2) SMOTE + Model (ONLY on training data)
    if args.use_smote:
        smote_model = ImbPipeline(steps=[
            ("smote", SMOTE(random_state=42)),
            ("rf", RandomForestClassifier(random_state=42))
        ])

        smote_model.fit(X_train, y_train)

        smote_preds = smote_model.predict(X_test)
        smote_probs = smote_model.predict_proba(X_test)[:, 1]
        smote_auc = roc_auc_score(y_test, smote_probs)

        print("\n=== SMOTE + RandomForest ===")
        print("Accuracy:", accuracy_score(y_test, smote_preds))
        print("Confusion Matrix:\n", confusion_matrix(y_test, smote_preds))
        print("Classification Report:\n", classification_report(y_test, smote_preds))
        print("ROC-AUC:", smote_auc)

        # Save the SMOTE pipeline as your main production model 
        model_path = ROOT / "models" / "random_forest_smote.pkl"
        joblib.dump(smote_model, model_path)
        print(f"\nSMOTE model saved to {model_path}")
    
if __name__ == "__main__":
    main()
