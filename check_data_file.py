import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

df = pd.read_csv("data_raw/datasets/official_data_en.csv")

print(df.head(3).to_string())
print(df.tail(3).to_string())
print("\nlevel:\n", df["level"].value_counts(dropna=False))
print("\nsource:\n", df["source"].value_counts(dropna=False))
print("\nnulls:\n", df.isna().sum())
print("\noblasts:", df["oblast"].nunique())
print(sorted(df["oblast"].dropna().unique()))

s = pd.to_datetime(df["started_at"], errors="coerce")
f = pd.to_datetime(df["finished_at"], errors="coerce")
print("\nstart min/max:", s.min(), s.max())
print("finish min/max:", f.min(), f.max())
print("unparsed start:", s.isna().sum(), " unparsed finish:", f.isna().sum())
print("sample raw started_at:", df["started_at"].iloc[[0, -1]].tolist())