# AI-Based Network Intrusion Detection for 5G/IoT Networks

A machine-learning prototype that classifies IoT network-traffic flows as **NORMAL** or **ATTACK**, and, for attacks, suggests the attack family. It runs entirely on a laptop, with a Flask API and web dashboard for demonstration.

---

## 1. Scope and honesty statement

Please read this before judging the project.

- **No physical 5G network and no hardware were used.** Nothing was captured from a real 5G core, gNB, or IoT device.
- The project is an **offline ML intrusion-detection prototype** trained on a public **IoT** traffic dataset (CICIoT2023). The pipeline (flow features, classifier, prediction service, dashboard) is *architecturally applicable* to IoT/5G environments, but **it has not been tested on 5G traffic**.
- Results come from one dataset captured in one lab testbed. They show the approach works on that data, **not** that the system would perform the same on a live network.
- The dashboard analyses **uploaded or sample feature rows**. It is not a live packet sniffer.

---

## 2. Dataset

- **Source:** CICIoT2023 (IoT Intrusion Detection), Kaggle version `IoT_Intrusion.csv`.
- **This file is an extract:** 1,048,575 rows x 47 columns (46 numeric features + `label`). The original CICIoT2023 collection is far larger.
- **Labels:** 34 classes (33 attacks + `BenignTraffic`), which we group into 8 families: Benign, DDoS, DoS, Mirai, Recon, Spoofing, Web, BruteForce.
- **No IP addresses or port numbers** exist in this file. Features are flow-level and packet-statistics based (durations, rates, TCP flag counts, protocol indicators, packet-size statistics, etc.). We did not invent any features.

### Preprocessing
1. No missing values were found. All features were already numeric.
2. **104,627 exact duplicate rows removed** (one copy kept) -> 943,948 rows. Done *before* splitting to avoid train/test leakage.
3. **5 constant features dropped:** `ece_flag_number`, `cwr_flag_number`, `Telnet`, `SMTP`, `IRC` -> **41 features** used.
4. **Subset:** each of the 34 classes capped at 20,000 rows -> 322,039 rows (302,039 attack, 20,000 normal).
5. **Split:** 80/20 stratified by family -> 257,631 train / 64,408 test (4,000 normal, 60,408 attack in test).
6. No feature scaling for tree models. Logistic Regression uses `StandardScaler` inside a pipeline fitted on training data only.

Correlated features (`Rate`/`Srate`, `IPv`/`LLC`, `Std`/`Radius`, `Number`/`Weight`/`IAT`) were kept, since Random Forest tolerates redundancy.

---

## 3. Results (held-out test set)

### Binary detector (Random Forest, Normal vs Attack)

| Metric | Value |
|---|---|
| Accuracy | 99.09% |
| Precision (Attack) | 99.54% |
| Recall (Attack) | 99.49% |
| F1 (Attack) | 99.52% |
| ROC-AUC | 0.9988 |
| False positive rate | 6.90% |

Confusion matrix (rows = actual, columns = predicted):

|  | Pred Normal | Pred Attack |
|---|---|---|
| **Actual Normal** | 3,724 | 276 |
| **Actual Attack** | 307 | 60,101 |

The **Normal class** scores precision 0.924 / recall 0.931. Overall accuracy is flattered by the fact that about 94% of the test set is attack traffic.

### Model comparison (binary)

| Model | Accuracy | F1 | ROC-AUC | False positive rate |
|---|---|---|---|---|
| Logistic Regression | 0.9496 | 0.9724 | 0.9860 | 1.70% |
| Naive Bayes | 0.9445 | 0.9699 | 0.9614 | 20.25% |
| Decision Tree | 0.9886 | 0.9939 | 0.9479* | 9.85% |
| **Random Forest (deployed)** | **0.9909** | **0.9952** | **0.9988** | 6.90% |
| Gradient Boosting | 0.9868 | 0.9929 | 0.9988 | **0.95%** |

\*A single decision tree outputs hard 0/1 probabilities, so its ROC-AUC is not a fair ranking measure.

Random Forest was chosen as the deployed model **before** the comparison was run, not selected from test-set scores. Gradient Boosting produced far fewer false alarms at slightly lower recall, which is a real trade-off and a candidate for future work.

### Attack-family classifier (Random Forest, 8 classes)

Accuracy 98.41%, **macro F1 0.84**. DDoS, DoS and Mirai reach about 0.999 F1. Rare classes are weak: **Web** recall 0.27 (108 test rows) and **BruteForce** recall 0.52 (65 test rows), with Recon and Spoofing around 0.82 to 0.86 F1. This is a bonus feature with a known limitation, not the headline result.

---

## 4. Limitations

1. **Not 5G data.** The dataset is IoT traffic from a lab testbed.
2. **Same-source evaluation.** Train and test come from the same capture. Performance on other networks would likely be lower.
3. **Near-duplicates remain.** Only exact duplicates were removed. Flood attacks produce many near-identical flows, which makes them easy to separate and inflates scores.
4. **Class capping changes base rates.** Normal is about 6% of the subset. Real networks are mostly normal, so real-world precision would be lower and false alarms more costly.
5. **6.9% false positive rate** is too high for unattended production use.
6. **Rare attack families** (Web, BruteForce) are poorly detected at the family level.
7. **Feature reliance.** Feature importance is led by `rst_count` and `urg_count`. This describes this dataset's traffic and may not generalize.
8. **Offline only.** Features must already be extracted. There is no live capture or feature-extraction pipeline.
9. **Dashboard statistics are kept in memory** and reset when the server restarts. Flask runs in debug mode, which is for development only.

## 5. Future work

Evaluate on a different dataset or capture; tune the decision threshold or use Gradient Boosting to reduce false positives; address rare-class imbalance (resampling or more data); add live flow extraction; test on 5G-specific traffic.

---

## 6. Project structure

```
ai-network-intrusion-detection/
├── dataset/            IoT_Intrusion.csv, subset.csv, demo_traffic.csv
├── models/             rf_binary.joblib, rf_family.joblib, features.json,
│                       metrics.json, model_comparison.json/.csv, feature_importance.csv
├── templates/          index.html
├── static/             style.css, script.js
├── notebooks/
├── inspect_dataset.py  inspect the raw CSV
├── preprocessing.py    clean, deduplicate, build subset
├── feature_analysis.py constant/correlation/importance analysis
├── train_model.py      train + evaluate binary and family models
├── compare_models.py   compare 5 classifiers
├── predict.py          prediction logic (single row and CSV)
├── app.py              Flask API + dashboard server
└── requirements.txt
```

## 7. How to run

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt

# Place IoT_Intrusion.csv in dataset/, then run in order:
python preprocessing.py
python feature_analysis.py
python train_model.py
python compare_models.py
python predict.py                # quick self-test

python app.py                    # open http://127.0.0.1:5000
```

Saved models are tied to the scikit-learn version used for training. Use the pinned versions in `requirements.txt`.

## 8. API

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/metrics` | GET | Model scores, confusion matrices, model comparison |
| `/api/stats` | GET | Running totals since server start |
| `/api/sample?type=normal\|attack` | GET | Example feature row |
| `/api/features` | GET | The 41 feature names |
| `/api/predict` | POST JSON `{"features": {...}}` | Predict one row |
| `/api/upload` | POST form-data, field `file` | Predict a CSV (first 200 rows returned) |
| `/api/reset` | POST | Reset running totals |

Uploaded CSVs must contain all 41 feature columns. An optional `label` column enables an accuracy report.

## 9. Credits

Dataset: CICIoT2023, Canadian Institute for Cybersecurity, University of New Brunswick. Check the dataset page for the full citation and licence terms before submission.