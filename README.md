Project Overview

This project uses machine learning to detect whether a person has Parkinson’s disease based on voice measurements.

Parkinson’s disease affects motor control, and subtle changes in a person’s voice can be measured numerically.
Using these measurements, we train a machine learning model to predict whether a person is healthy or has Parkinson’s disease.

The Data Set:
File: data/parkinsons.data
Each row in the dataset represents one voice recording from a person.

Important columns:
Voice features → numeric measurements extracted from speech

status (label):
0 → Healthy
1 → Parkinson’s disease

This column is what the model tries to predict.

1st Step:
Data Preparation (inside train.py)
What happens

The dataset is loaded

The label (status) is separated from the features

Non-numeric columns (like IDs) are removed

Why this is needed

Machine learning models can only learn from numbers.
Separating features and labels is required so the model knows:

what to learn from (features)

what to predict (label)

2nd Step:
Train / Test Split
What happens

The dataset is split into:

80% training data

20% testing data

Why this is needed

The model must be evaluated on data it has never seen before, to simulate real-world patients.

We also use:

stratify=y → keeps the same ratio of healthy vs Parkinson’s in both sets

random_state=42 → makes results reproducible


3rd Step:
Baseline Model: RandomForest (No SMOTE)
What is RandomForest?

RandomForest is an ensemble model that:

builds many decision trees

combines their predictions

handles non-linear patterns well

It is widely used in medical and healthcare ML.

Why we start with a baseline

A baseline answers:

“How well does the model perform naturally, without extra techniques?”

This gives us something fair to compare against later.

Output produced

Accuracy

Confusion Matrix

Precision, Recall, F1-score

ROC-AUC score

Saved model: models/random_forest.pkl


4th Step:
Understanding the Confusion Matrix

Example:

[[ 8  2]
 [ 1 28]]


This means:

TN (True Negative) = 8 → Healthy correctly predicted

FP (False Positive) = 2 → Healthy predicted as Parkinson’s

FN (False Negative) = 1 → Parkinson’s predicted as healthy **

TP (True Positive) = 28 → Parkinson’s correctly predicted

Why FN is critical

In medicine:

Missing a sick patient (FN) is worse than a false alarm (FP)

So we prioritize recall, not just accuracy.


5th Step:
SMOTE Experiment (Class Imbalance Handling)
The problem

The dataset has more Parkinson’s cases than healthy cases.

This imbalance can bias the model.

What SMOTE does

SMOTE (Synthetic Minority Oversampling Technique):

1.creates synthetic samples of the minority class

2.applied only to training data

3.helps the model learn balanced patterns

Why we tested it because we wanted to see if SMOTE:

1.reduces false negatives

2.improves recall

3.improves ROC-AUC

4.Result

SMOTE slightly improved ROC-AUC but did not reduce false negatives compared to baseline.

This shows why testing is more important than blindly using techniques.


6th Step:
ROC Curve & ROC-AUC (evaluate.py)
What ROC shows

The ROC curve shows how well the model separates:

Healthy patients

Parkinson’s patients

Across all possible probability thresholds.

ROC-AUC value

0.5 → random guessing

1.0 → perfect model

Our model:

ROC-AUC ≈ 0.96

This means the model is very good at ranking risk, not just predicting labels.

The ROC curve is saved as:

reports/roc_curve.png


7th Step:
Threshold Tuning (threshold_tuning.py)
Default behavior

Most models use:

probability ≥ 0.5 → Parkinson’s


But this may not be optimal for medical decisions.

What we did

We tested thresholds from 0.1 to 0.9 and measured:

TN, FP, FN, TP

Recall

Precision

Why threshold = 0.3 was chosen

At threshold 0.3:

Very high recall (almost all Parkinson’s detected)

Acceptable precision

Fewer missed patients

This reflects real medical decision-making.


8th Step:
Hyperparameter Tuning (tune.py)
What this does

GridSearchCV tries many RandomForest configurations using:

5-fold cross-validation

recall as the optimization metric

Why recall was optimized

Because:

Missing a Parkinson’s patient is more dangerous than a false alarm

Result

The tuned model performed similarly to the baseline on the test set.

This confirmed that the simpler baseline model was already optimal.

This is a valid and important ML conclusion.


9th Step:
Prediction on New Data (predict.py)

This script:

loads the trained model

loads feature order

applies the chosen threshold

predicts for a single row

Example output:

Predicted: 1 (0=Healthy, 1=Parkinson’s)
Probability of Parkinson’s: 0.99
True Label: 1


This simulates real-world deployment logic.

Decisions:
Final Model Choice after comparing:

1.baseline

2.SMOTE

3.tuned models

We selected:

Model: RandomForest (baseline)

Threshold: 0.3

Metric priority: Recall

Saved model: models/random_forest.pkl

This choice is based on medical risk, not just accuracy.

What This Project Demonstrates:

End-to-end ML pipeline

Correct evaluation metrics

Medical decision reasoning

Model comparison

Threshold tuning

Reproducible experiments

Clean project structure

How to run:
pip install -r requirements.txt

python -m src.train
python -m src.evaluate
python -m src.threshold_tuning
python -m src.tune
python -m src.predict --row 0
