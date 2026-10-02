import json

import joblib
import numpy as np
import pandas as pd

with open("models/features.json") as f:
    FEATURES = json.load(f)

_bin = joblib.load("models/rf_binary.joblib")
_fam = joblib.load("models/rf_family.joblib")


def _prepare(df: pd.DataFrame):
    """Check columns, coerce to numeric, drop unusable rows."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} required column(s): {missing[:8]}"
            + (" ..." if len(missing) > 8 else "")
        )
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    before = len(X)
    X = X.dropna()
    return X, before - len(X)


def predict_dataframe(df: pd.DataFrame, threshold: float = 0.5):
    """Returns (results_df, n_rows_dropped). results keep the input row index."""
    X, dropped = _prepare(df)
    if len(X) == 0:
        raise ValueError("No valid rows to predict.")

    prob = _bin.predict_proba(X)[:, 1]
    is_attack = prob >= threshold

    # Attack family: most likely family, excluding 'Benign'
    fam_proba = _fam.predict_proba(X)
    classes = list(_fam.classes_)
    fam_proba[:, classes.index("Benign")] = 0.0
    fam_pred = np.array(classes)[fam_proba.argmax(axis=1)]

    out = pd.DataFrame(index=X.index)
    out["prediction"] = np.where(is_attack, "ATTACK", "NORMAL")
    out["attack_probability"] = prob.round(4)
    out["attack_family"] = np.where(is_attack, fam_pred, "-")
    return out, dropped


def predict_one(values: dict) -> dict:
    """values: {feature_name: number}. Missing features default to 0."""
    row = pd.DataFrame([{f: values.get(f, 0) for f in FEATURES}])
    out, _ = predict_dataframe(row)
    r = out.iloc[0]
    return {
        "prediction": str(r["prediction"]),
        "attack_probability": float(r["attack_probability"]),
        "attack_family": str(r["attack_family"]),
    }


def summarize(results: pd.DataFrame) -> dict:
    attacks = results[results["prediction"] == "ATTACK"]
    return {
        "total": int(len(results)),
        "normal": int((results["prediction"] == "NORMAL").sum()),
        "attack": int(len(attacks)),
        "family_counts": attacks["attack_family"].value_counts().to_dict(),
    }


if __name__ == "__main__":
    demo = pd.read_csv("dataset/demo_traffic.csv")
    res, dropped = predict_dataframe(demo)
    print(f"Rows predicted: {len(res)}  (dropped: {dropped})")
    print("Summary:", summarize(res))

    actual = np.where(demo.loc[res.index, "family"] == "Benign", "NORMAL", "ATTACK")
    print("\nAccuracy on demo file:", round((res["prediction"].values == actual).mean(), 4))
    print("\nActual (rows) vs predicted (columns):")
    print(pd.crosstab(pd.Series(actual, name="actual"), res["prediction"].reset_index(drop=True)))

    print("\nSingle-row test (one normal row, one attack row from the demo file):")
    for fam in ["Benign", "DDoS"]:
        sample = demo[demo["family"] == fam].iloc[0]
        print(f"  true={sample['label']:<18} ->", predict_one(sample[FEATURES].to_dict()))