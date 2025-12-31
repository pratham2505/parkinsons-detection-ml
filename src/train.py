from pathlib import Path
import pandas as pd # loads datasets file into a table-like structure
import joblib # saves and loads trained model files
from sklearn.model_selection import train_test_split # splits datasets into training and testing sets
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # different ways to measure how good my model is
from sklearn.ensemble import RandomForestClassifier # the model that we are training
from sklearn.metrics import roc_auc_score
import json

ROOT = Path(__file__).resolve().parents[1]

def main():
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

    model = RandomForestClassifier(random_state=42) # create the model by building many decision trees and averaging their results
    model.fit(X_train, y_train) # now we train the model on the training data

    model_path = ROOT / "models" / "random_forest.pkl"
    joblib.dump(model, model_path) # save the trained model to a file so we
    print(f"Model saved to {model_path}")

    preds = model.predict(X_test) # use the trained model to make predictions on the test data

    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))

    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    print("ROC-AUC Score:", auc)
    
if __name__ == "__main__":
    main()
