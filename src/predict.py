from pathlib import Path
import pandas as pd
import argparse
import joblib
import json

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser() # create argument parser
    parser.add_argument("--row", type=int, default=0, help="Row index to predict from dataset") # add argument for row index because we want to specify which row to predict
    args = parser.parse_args() # parse the command line arguments

    model_path = ROOT / "models" / "random_forest.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Please train the model first.")  
    
    model = joblib.load(model_path) # load the trained model

    df = pd.read_csv(ROOT / "data" / "parkinsons.data") # read the dataset
    
    # 4) Load the exact feature columns used during training (and keep the same order)
    cols_path = ROOT / "models" / "feature_columns.json"
    if not cols_path.exists():
        raise FileNotFoundError(f"Feature columns file not found at {cols_path}. Run: python -m src.train")

    with open(cols_path, "r") as f:
        feature_cols = json.load(f)

    # 5) Build X using those exact columns
    # This ensures prediction never breaks due to column order changes.
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required feature columns: {missing}")

    X = df[feature_cols]
    y = df["status"] if "status" in df.columns else None

    if args.row < 0 or args.row >= len(X):
        raise ValueError(f"--row must be between 0 and {len(X)-1}")
    
    x_one = X.iloc[[args.row]] # get the specified row as a dataframe
    THRESHOLD = 0.3

    prob = model.predict_proba(x_one)[0, 1]
    pred = int(prob >= THRESHOLD)
    
    # prob = model.predict_proba(x_one)[0,1] # get probability of having parkinsons
    # pred = int(model.predict(x_one)[0]) # get the predicted class (0 or 1)

    print(f"Row {args.row}")
    print(f"Predicted: {pred} (0=Healthy, 1=Parkinson's)")
    print(f"Prob of Parkinson's: {prob:.4f}")
    print(f"Decision threshold: {THRESHOLD}")
    if y is not None:
        true_label = int(y.iloc[args.row])
        print(f"True Label: {true_label}")

if __name__ == "__main__":
    main()
    