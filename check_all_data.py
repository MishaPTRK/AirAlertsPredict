import pandas as pd

for name in ["official_data_en", "official_data_uk", "volunteer_data_en", "volunteer_data_uk"]:
    df = pd.read_csv(f"data_raw/datasets/{name}.csv")
    col = [c for c in df.columns if "start" in c.lower()][0]
    s = pd.to_datetime(df[col], errors="coerce", utc=True)
    print(name, len(df), "|", col, s.min(), "->", s.max())