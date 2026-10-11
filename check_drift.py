import pandas as pd
from src.grid import to_hourly

v = pd.read_csv("data_raw/datasets/volunteer_data_en.csv")
for c in ["started_at", "finished_at"]:
    v[c] = pd.to_datetime(v[c], utc=True)
v = v[v["region"] != "Luhanska oblast"]

# 1) довгі тривоги по областях
v["hours"] = (v["finished_at"] - v["started_at"]).dt.total_seconds() / 3600
long = v[v["hours"] > 24]
print("alerts >24h:", len(long), "| >12h:", (v["hours"] > 12).sum())
print(long.groupby("region").size().sort_values(ascending=False).to_string())

# 2) місячна частка активних годин (середнє по областях), чи є злам близько грудня 2025
grid = to_hourly(v, "region", "2025-01-01", "2026-10-11")
monthly = grid.groupby(grid.index.tz_localize(None).to_period("M")).mean()
print(monthly.mean(axis=1).round(3).to_string())

# 3) те саме для кількох ключових областей
print(monthly[["Kharkivska oblast", "Dnipropetrovska oblast",
               "Kyiv City", "Lvivska oblast"]].round(3).to_string())