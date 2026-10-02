import json
import os
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

RANDOM_STATE = 42
os.makedirs("models", exist_ok=True)

df = pd.read_csv("dataset/subset.csv")
with open("models/features.json") as f:
    FEATURES = json.load(f)

train, test = train_test_split(
    df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["family"]
)
X_train, X_test = train[FEATURES], test[FEATURES]
y_train, y_test = train["is_attack"].values, test["is_attack"].values

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    ),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced", random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
    ),
    "Gradient Boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
}

# HistGradientBoosting gets class balancing through sample weights
sw = compute_sample_weight("balanced", y_train)

rows = []
for name, model in models.items():
    print(f"Training {name}...")
    t0 = time.time()
    if name == "Gradient Boosting":
        model.fit(X_train, y_train, sample_weight=sw)
    else:
        model.fit(X_train, y_train)
    train_s = time.time() - t0

    t0 = time.time()
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    pred_s = time.time() - t0

    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    rows.append({
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, prob),
        "false_positive_rate": fp / (fp + tn),
        "train_seconds": round(train_s, 1),
        "predict_seconds": round(pred_s, 2),
    })

res = pd.DataFrame(rows).set_index("model")
pd.set_option("display.width", 200)
print("\n=== MODEL COMPARISON (binary: Normal vs Attack, test set) ===")
print(res.round(4).to_string())

res.round(4).to_csv("models/model_comparison.csv")
with open("models/model_comparison.json", "w") as f:
    json.dump(rows, f, indent=2, default=float)
print("\nSaved models/model_comparison.csv and models/model_comparison.json")