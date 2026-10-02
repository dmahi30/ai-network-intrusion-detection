import glob
import os
import pandas as pd

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 200)

# Find every CSV inside dataset/
files = sorted(glob.glob("dataset/**/*.csv", recursive=True))

print(f"Found {len(files)} CSV file(s)\n")

for f in files[:15]:
    size_mb = os.path.getsize(f) / (1024 * 1024)
    print(f"{size_mb:8.1f} MB  {f}")

if len(files) > 15:
    print(f"... and {len(files) - 15} more")

if not files:
    raise SystemExit(
        "No CSV files found. Check that the dataset was extracted into dataset/"
    )

# Load only first 200,000 rows
first = files[0]

print(f"\nInspecting: {first}")

df = pd.read_csv(first, nrows=200_000)

print("\nShape:", df.shape)

print("\nColumns and dtypes:")
print(df.dtypes.to_string())

print("\nFirst 3 rows:")
print(df.head(3))

print("\nMissing values (total):", int(df.isnull().sum().sum()))

print("Duplicate rows:", int(df.duplicated().sum()))

# Find possible label column
label_cols = [
    c for c in df.columns
    if c.lower() in ("label", "class", "attack", "attack_type")
]

print("\nPossible label column(s):", label_cols)

if label_cols:
    print("\nLabel distribution:")
    print(df[label_cols[0]].value_counts().to_string())