import pandas as pd

v = pd.read_csv("data_raw/datasets/volunteer_data_en.csv")
for c in ["started_at", "finished_at"]:
    v[c] = pd.to_datetime(v[c], utc=True)

print("regions:", v["region"].nunique())
print(sorted(v["region"].unique()))
print("naive share overall:", round(v["naive"].mean(), 4))

v["year"] = v["started_at"].dt.year
print("\nnaive share by year:\n", v.groupby("year")["naive"].mean().round(4))

dur = (v["finished_at"] - v["started_at"])
print("\nnegative durations:", (dur < pd.Timedelta(0)).sum())
print("duration quantiles (min):\n", (dur.dt.total_seconds() / 60).describe(percentiles=[.5, .9, .99]))

# порівняння з офіційним, період, коли в офіційних ще були області
o = pd.read_csv("data_raw/datasets/official_data_en.csv")
o = o[o["level"] == "oblast"].rename(columns={"oblast": "region"})
o["started_at"] = pd.to_datetime(o["started_at"], utc=True)
print("\nofficial oblast-level max date:", o["started_at"].max())

lo, hi = pd.Timestamp("2025-06-01", tz="UTC"), pd.Timestamp("2025-11-30", tz="UTC")
vc = v[v["started_at"].between(lo, hi)].groupby("region").size()
oc = o[o["started_at"].between(lo, hi)].groupby("region").size()
cmp = pd.concat([vc.rename("volunteer"), oc.rename("official")], axis=1)
cmp["ratio"] = (cmp["volunteer"] / cmp["official"]).round(2)
print(cmp.to_string())