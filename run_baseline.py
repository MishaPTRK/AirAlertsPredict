import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from src.data import load_alerts, build_panel, make_long

TEST_START = pd.Timestamp("2026-09-13", tz="UTC")
TEST_END = pd.Timestamp("2026-10-11", tz="UTC")  # виключно: 11.10 неповна доба
HISTORY_WEEKS = 26
EXCLUDE = ["Volynska oblast", "Lvivska oblast"]


def run(df, label):
    train = df[df["time"] < TEST_START]
    test = df[(df["time"] >= TEST_START) & (df["time"] < TEST_END)].copy()
    recent = train[train["time"] >= TEST_START - pd.Timedelta(weeks=HISTORY_WEEKS)]
    base = recent["y"].mean()
    how_mean = recent.groupby(["region", "how"])["y"].mean().rename("how_mean")
    test = test.join(how_mean, on=["region", "how"])
    preds = {
        "persistence (lag1)": test["lag1"],
        "active_now": test["now"],
        "freq_24h": test["freq24h"],
        "freq_7d": test["freq7d"],
        "hour_of_week_mean": test["how_mean"].fillna(base),
        "constant (base rate)": pd.Series(base, index=test.index),
    }
    print(f"\n=== {label} | test rows: {len(test)} | train base: {base:.4f} "
          f"| test base: {test['y'].mean():.4f}")
    rows = [{"baseline": n, "PR-AUC": average_precision_score(test["y"], p),
             "Brier": brier_score_loss(test["y"], p)} for n, p in preds.items()]
    print(pd.DataFrame(rows).round(4).to_string(index=False))


v = load_alerts()
df = make_long(build_panel(v, "2025-01-01", "2026-10-11"))
df = df.dropna(subset=["lag1", "lag24", "freq24h", "freq7d"])
print("share masked as bad:", df["bad"].mean().round(4))

run(df[~df["bad"]], "masked long alerts (main)")
run(df, "NOT masked")
run(df[~df["bad"] & ~df["region"].isin(EXCLUDE)], "masked, without Volyn + Lviv")