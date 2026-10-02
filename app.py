import json

import pandas as pd
from flask import Flask, jsonify, render_template, request

import predict as P

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB upload limit

with open("models/metrics.json") as f:
    METRICS = json.load(f)
try:
    with open("models/model_comparison.json") as f:
        COMPARISON = json.load(f)
except FileNotFoundError:
    COMPARISON = []

DEMO = pd.read_csv("dataset/demo_traffic.csv")

# Running totals since the server started (reset with POST /api/reset)
STATS = {"total": 0, "normal": 0, "attack": 0, "family_counts": {}}


def _add_to_stats(summary: dict):
    STATS["total"] += summary["total"]
    STATS["normal"] += summary["normal"]
    STATS["attack"] += summary["attack"]
    for fam, n in summary["family_counts"].items():
        STATS["family_counts"][fam] = STATS["family_counts"].get(fam, 0) + n


@app.errorhandler(ValueError)
def bad_request(e):
    return jsonify({"error": str(e)}), 400


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large (limit 50 MB)."}), 413


# ---------- Pages ----------
@app.route("/")
def index():
    return render_template("index.html")


# ---------- API ----------
@app.route("/api/metrics")
def api_metrics():
    return jsonify({**METRICS, "comparison": COMPARISON})


@app.route("/api/features")
def api_features():
    return jsonify({"features": P.FEATURES})


@app.route("/api/sample")
def api_sample():
    """Random example row to prefill the single-prediction form. ?type=normal|attack"""
    kind = request.args.get("type", "attack")
    pool = DEMO[DEMO["family"] == "Benign"] if kind == "normal" else DEMO[DEMO["family"] != "Benign"]
    row = pool.sample(1).iloc[0]
    return jsonify({
        "features": {f: float(row[f]) for f in P.FEATURES},
        "true_label": str(row["label"]),
        "true_family": str(row["family"]),
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    feats = data.get("features")
    if not isinstance(feats, dict):
        raise ValueError('Send JSON like {"features": {"rst_count": 0, ...}}')
    result = P.predict_one(feats)
    _add_to_stats({
        "total": 1,
        "normal": int(result["prediction"] == "NORMAL"),
        "attack": int(result["prediction"] == "ATTACK"),
        "family_counts": {} if result["prediction"] == "NORMAL" else {result["attack_family"]: 1},
    })
    return jsonify(result)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    file = request.files.get("file")
    if file is None or file.filename == "":
        raise ValueError("No file uploaded. Send a CSV in the form field 'file'.")
    try:
        df = pd.read_csv(file)
    except Exception:
        raise ValueError("Could not read the file as CSV.")

    results, dropped = P.predict_dataframe(df)
    summary = P.summarize(results)
    summary["dropped_rows"] = dropped
    _add_to_stats(summary)

    # If the file has a 'label' column, report accuracy against it
    evaluation = None
    if "label" in df.columns:
        actual_attack = (df.loc[results.index, "label"].astype(str).str.strip() != "BenignTraffic").values
        pred_attack = (results["prediction"] == "ATTACK").values
        evaluation = {
            "accuracy": round(float((actual_attack == pred_attack).mean()), 4),
            "true_normal_pred_normal": int(((~actual_attack) & (~pred_attack)).sum()),
            "true_normal_pred_attack": int(((~actual_attack) & pred_attack).sum()),
            "true_attack_pred_normal": int((actual_attack & (~pred_attack)).sum()),
            "true_attack_pred_attack": int((actual_attack & pred_attack).sum()),
        }

    preview = results.head(200).reset_index(drop=True)
    preview.insert(0, "row", range(1, len(preview) + 1))
    return jsonify({
        "summary": summary,
        "evaluation": evaluation,
        "rows": preview.to_dict(orient="records"),
        "rows_shown": len(preview),
    })


@app.route("/api/stats")
def api_stats():
    return jsonify(STATS)


@app.route("/api/reset", methods=["POST"])
def api_reset():
    STATS.update({"total": 0, "normal": 0, "attack": 0, "family_counts": {}})
    return jsonify(STATS)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)