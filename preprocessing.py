import os
import pandas as pd

RAW_PATH = "dataset/IoT_Intrusion.csv"
SUBSET_PATH = "dataset/subset.csv"
CAP_PER_CLASS = 20_000
RANDOM_STATE = 42

# Grouping of the 34 fine-grained labels into 8 families
# (follows the families described by the CICIoT2023 authors)
def to_family(label: str) -> str:
    if label == "BenignTraffic":
        return "Benign"
    if label.startswith("DDoS"):
        return "DDoS"
    if label.startswith("DoS"):
        return "DoS"
    if label.startswith("Mirai"):
        return "Mirai"
    if label.startswith("Recon") or label == "VulnerabilityScan":
        return "Recon"
    if label in ("MITM-ArpSpoofing", "DNS_Spoofing"):
        return "Spoofing"
    if label == "DictionaryBruteForce":
        return "BruteForce"
    # SqlInjection, XSS, CommandInjection, BrowserHijacking,
    # Uploading_Attack, Backdoor_Malware
    return "Web"


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # float32 halves memory use; precision is plenty for this task
    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].astype("float32")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().reset_index(drop=True)
    df["label"] = df["label"].str.strip()
    df["is_attack"] = (df["label"] != "BenignTraffic").astype(int)
    df["family"] = df["label"].map(to_family)
    return df


def make_subset(df: pd.DataFrame, cap: int = CAP_PER_CLASS) -> pd.DataFrame:
    parts = []
    for _, g in df.groupby("label"):
        parts.append(g.sample(n=min(len(g), cap), random_state=RANDOM_STATE))
    sub = pd.concat(parts).sample(frac=1, random_state=RANDOM_STATE)
    return sub.reset_index(drop=True)


if __name__ == "__main__":
    raw = load_raw()
    print("Raw shape:", raw.shape)
    print("Missing values:", int(raw.isnull().sum().sum()))
    print("Duplicates:", int(raw.duplicated().sum()))

    df = clean(raw)
    print("\nAfter dropping duplicates:", df.shape)

    print("\nBinary distribution (full cleaned data):")
    print(df["is_attack"].value_counts().rename({0: "Normal", 1: "Attack"}).to_string())

    print("\nFamily distribution (full cleaned data):")
    print(df["family"].value_counts().to_string())

    # Near-constant feature check
    feats = df.drop(columns=["label", "is_attack", "family"])
    nun = feats.nunique().sort_values()
    print("\nFeatures with <= 3 unique values:")
    print(nun[nun <= 3].to_string())

    sub = make_subset(df)
    os.makedirs("dataset", exist_ok=True)
    sub.to_csv(SUBSET_PATH, index=False)
    print(f"\nSubset saved to {SUBSET_PATH} with shape {sub.shape}")
    print("\nSubset binary distribution:")
    print(sub["is_attack"].value_counts().rename({0: "Normal", 1: "Attack"}).to_string())
    print("\nSubset family distribution:")
    print(sub["family"].value_counts().to_string())