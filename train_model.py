import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split

os.makedirs("models", exist_ok=True)
RANDOM_STATE = 42

# ---------- Load ----------
df = pd.read_csv("dataset/subset.csv")
with open("models/features.json") as f:
    FEATURES = json.load(f)

# Same split as feature_analysis.py (same seed + stratify column)
train, test = train_test_split(
    df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["family"]
)
X_train, X_test = train[FEATURES], test[FEATURES]
print(f"Train: {X_train.shape}  Test: {X_test.shape}")

# ---------- 1. Binary model: Normal (0) vs Attack (1) ----------
print("\nTraining binary model (Normal vs Attack)...")
t0 = time.time()
rf_bin = RandomForestClassifier(
    n_estimators=100,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    class_weight="balanced_subsample",
)
rf_bin.fit(X_train, train["is_attack"])
print(f"  done in {time.time() - t0:.0f}s")

y_true = test["is_attack"].values
y_pred = rf_bin.predict(X_test)
y_prob = rf_bin.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # rows=actual, cols=predicted
tn, fp, fn, tp = cm.ravel()

binary_metrics = {
    "accuracy": accuracy_score(y_true, y_pred),
    "precision": precision_score(y_true, y_pred),   # Attack = positive class
    "recall": recall_score(y_true, y_pred),
    "f1": f1_score(y_true, y_pred),
    "roc_auc": roc_auc_score(y_true, y_prob),
    "false_positive_rate": fp / (fp + tn),
    "confusion_matrix": cm.tolist(),
    "labels": ["Normal", "Attack"],
}

print("\n=== BINARY RESULTS (test set) ===")
for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate"]:
    print(f"{k:>20}: {binary_metrics[k]:.4f}")
print("\nConfusion matrix (rows = actual, cols = predicted) [Normal, Attack]:")
print(cm)
print("\n", classification_report(y_true, y_pred, target_names=["Normal", "Attack"], digits=4))

# ---------- 2. Family model (optional multi-class) ----------
print("Training family model (attack category)...")
t0 = time.time()
rf_fam = RandomForestClassifier(
    n_estimators=100,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    class_weight="balanced_subsample",
)
rf_fam.fit(X_train, train["family"])
print(f"  done in {time.time() - t0:.0f}s")

fam_pred = rf_fam.predict(X_test)
fam_true = test["family"].values
fam_labels = sorted(df["family"].unique())
fam_cm = confusion_matrix(fam_true, fam_pred, labels=fam_labels)

family_metrics = {
    "accuracy": accuracy_score(fam_true, fam_pred),
    "macro_f1": f1_score(fam_true, fam_pred, average="macro"),
    "weighted_f1": f1_score(fam_true, fam_pred, average="weighted"),
    "confusion_matrix": fam_cm.tolist(),
    "labels": fam_labels,
}

print("\n=== FAMILY RESULTS (test set) ===")
print(f"accuracy: {family_metrics['accuracy']:.4f}   macro F1: {family_metrics['macro_f1']:.4f}")
print("\n", classification_report(fam_true, fam_pred, digits=4))

# ---------- 3. Save everything ----------
joblib.dump(rf_bin, "models/rf_binary.joblib", compress=3)
joblib.dump(rf_fam, "models/rf_family.joblib", compress=3)

metrics = {
    "train_rows": int(len(train)),
    "test_rows": int(len(test)),
    "n_features": len(FEATURES),
    "test_normal": int((y_true == 0).sum()),
    "test_attack": int((y_true == 1).sum()),
    "binary": binary_metrics,
    "family": family_metrics,
    "test_family_counts": test["family"].value_counts().to_dict(),
    "top_features": dict(
        pd.Series(rf_bin.feature_importances_, index=FEATURES)
        .sort_values(ascending=False).head(10).round(4)
    ),
}
with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2, default=float)

# Small labeled demo file for the upload feature (500 normal + 500 attack from TEST set)
demo = pd.concat([
    test[test["is_attack"] == 0].sample(500, random_state=1),
    test[test["is_attack"] == 1].sample(500, random_state=1),
]).sample(frac=1, random_state=1)
demo[FEATURES + ["label", "family"]].to_csv("dataset/demo_traffic.csv", index=False)

print("\nSaved: models/rf_binary.joblib, models/rf_family.joblib, models/metrics.json")
print("Saved: dataset/demo_traffic.csv (1000 unseen test rows for demo uploads)")
for p in ["models/rf_binary.joblib", "models/rf_family.joblib"]:
    print(f"  {p}: {os.path.getsize(p) / 1e6:.1f} MB")