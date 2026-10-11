import pandas as pd
from src.grid import to_hourly

v = pd.read_csv("data_raw/datasets/volunteer_data_en.csv")
o = pd.read_csv("data_raw/datasets/official_data_en.csv")
o = o[o["level"] == "oblast"].rename(columns={"oblast": "region"})
for d in (v, o):
    d["started_at"] = pd.to_datetime(d["started_at"], utc=True)
    d["finished_at"] = pd.to_datetime(d["finished_at"], utc=True)

# 1) довгі тривоги
dur_h = (v["finished_at"] - v["started_at"]).dt.total_seconds() / 3600
print(v[dur_h > 24].assign(hours=dur_h).sort_values("hours", ascending=False).head(10).to_string())

# 2) збіг годинної активності, 2025-06-01 .. 2025-12-01
lo, hi = "2025-06-01", "2025-12-01"
gv = to_hourly(v, "region", lo, hi)
go = to_hourly(o, "region", lo, hi)
rows = []
for r in gv.columns:
    a, b = gv[r], go[r]
    rows.append({
        "region": r,
        "vol_share": a.mean().round(3),
        "off_share": b.mean().round(3),
        "agree": (a == b).mean().round(3),
        "jaccard": ((a & b).sum() / max((a | b).sum(), 1)).round(3),
    })
print(pd.DataFrame(rows).to_string(index=False))