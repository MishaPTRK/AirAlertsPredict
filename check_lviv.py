import pandas as pd
from src.data import load_alerts
from src.grid import to_hourly

v = load_alerts()

# чутливість до порогу довгих тривог
for t in (12, 24, 48):
    print(f"alerts > {t}h:", int((v["hours"] > t).sum()))

o = pd.read_csv("data_raw/datasets/official_data_en.csv")
for c in ("started_at", "finished_at"):
    o[c] = pd.to_datetime(o[c], utc=True)
o = o.rename(columns={"oblast": "region"})  # усі рівні: область активна, якщо активний хоч район/громада
o = o[o["region"] != "Luhanska oblast"]

west = ["Lvivska oblast", "Volynska oblast", "Rivnenska oblast",
        "Ternopilska oblast", "Ivano-Frankivska oblast", "Zakarpatska oblast"]
start, end = "2025-06-01", "2026-09-06"
gv = to_hourly(v[v["region"].isin(west)], "region", start, end)
go = to_hourly(o[o["region"].isin(west)], "region", start, end)


def monthly(g):
    return g.groupby(g.index.tz_localize(None).to_period("M")).mean().round(3)


print("\nVOLUNTEER, share of active hours"); print(monthly(gv).to_string())
print("\nOFFICIAL (any level), share of active hours"); print(monthly(go).to_string())

l = v[(v["region"] == "Lvivska oblast") & (v["started_at"] >= "2025-06-01")].copy()
l["month"] = l["started_at"].dt.tz_localize(None).dt.to_period("M")
print("\nLviv volunteer alerts per month")
print(l.groupby("month").agg(alerts=("hours", "size"), mean_h=("hours", "mean"),
                             median_h=("hours", "median"), naive=("naive", "mean")).round(2).to_string())