import json
import numpy as np
import pandas as pd
import os
os.makedirs("models", exist_ok=True)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

pd.set_option("display.width", 200)

df = pd.read_csv("dataset/subset.csv")
meta = ["label", "is_attack", "family"]
all_feats = [c for c in df.columns if c not in meta]

# 1. Remove constant features
const = [c for c in all_feats if df[c].nunique() == 1]
feats = [c for c in all_feats if c not in const]
print("Constant features dropped:", const)
print("Candidate features remaining:", len(feats))

# 2. Conflicting rows (same features, different labels)
dup_mask = df.duplicated(subset=feats, keep=False)
conflicts = df[dup_mask].groupby(feats)["label"].nunique()
n_conf = int((conflicts > 1).sum())
print(f"\nFeature-identical row groups with conflicting labels: {n_conf}")

# 3. Split FIRST, analyse on train only
train, test = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["family"]
)
print("\nTrain:", train.shape, " Test:", test.shape)

# 4. Highly correlated pairs (train only)
corr = train[feats].corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
pairs = (
    upper.stack()
    .loc[lambda s: s > 0.95]
    .sort_values(ascending=False)
)
print(f"\nFeature pairs with |correlation| > 0.95: {len(pairs)}")
print(pairs.head(15).to_string())

# 5. Random Forest importance (train sample for speed)
sample = train.sample(n=min(100_000, len(train)), random_state=42)
rf = RandomForestClassifier(
    n_estimators=100, n_jobs=-1, random_state=42,
    class_weight="balanced_subsample",
)
rf.fit(sample[feats], sample["is_attack"])
imp = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=False)
print("\nTop 15 features by importance (binary Normal vs Attack):")
print(imp.head(15).round(4).to_string())
print("\nBottom 8 features:")
print(imp.tail(8).round(4).to_string())

# Save results
imp.to_csv("models/feature_importance.csv", header=["importance"])
with open("models/features.json", "w") as f:
    json.dump(feats, f, indent=2)
print("\nSaved models/feature_importance.csv and models/features.json")