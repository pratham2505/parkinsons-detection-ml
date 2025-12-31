from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]

def main():
    df = pd.read_csv(ROOT / "data" / "parkinsons.data")

    y = df["status"]
    X = df.drop(columns=["status"])

    # drop any non-numeric columns (like the 'name' ID)
    X = X.select_dtypes(exclude=["object"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))

if __name__ == "__main__":
    main()
