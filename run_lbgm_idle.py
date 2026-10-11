import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, brier_score_loss

from src.data import load_alerts, build_panel
from src.features import make_features

TEST_START = pd.Timestamp("2026-09-13", tz="UTC")
TEST_END = pd.Timestamp("2026-10-11", tz="UTC")
TRAIN_WEEKS = 26
FEATURES = ["now", "lag1", "lag2", "lag3", "lag24", "freq3h", "freq6h", "freq24h",
            "freq7d", "freq30d", "since_last", "others_now", "others_prev",
            "hour", "dow", "region_id"]

df = make_features(build_panel(load_alerts(), "2025-01-01", "2026-10-11"))
df = df.dropna(subset=["lag24", "freq30d"])
df = df[~df["bad"]]
idle = df[df["now"] == 0]  # лише ті, де тривоги зараз немає

params = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)

parts = []
for ws in pd.date_range(TEST_START, TEST_END, freq="7D", inclusive="left"):
    we = ws + pd.Timedelta(days=7)
    lo = ws - pd.Timedelta(weeks=TRAIN_WEEKS)
    train = idle[(idle["time"] < ws) & (idle["time"] >= lo)]
    test = idle[(idle["time"] >= ws) & (idle["time"] < we)].copy()
    m = LGBMClassifier(**params).fit(train[FEATURES], train["y"])
    test["p_idle"] = m.predict_proba(test[FEATURES])[:, 1]
    parts.append(test)
res = pd.concat(parts)

# модель, навчена на всіх рядках, для порівняння
old = pd.read_parquet("predictions_test.parquet")[["time", "region", "p"]]
res = res.merge(old, on=["time", "region"], how="left")

cols = {"LightGBM idle-only": "p_idle", "LightGBM all-rows": "p", "freq_7d": "freq7d"}
print("n =", len(res), "| base rate =", round(res["y"].mean(), 4))
print(pd.DataFrame([{"model": n, "PR-AUC": average_precision_score(res["y"], res[c]),
                     "Brier": brier_score_loss(res["y"], res[c])}
                    for n, c in cols.items()]).round(4).to_string(index=False))

# бутстреп по добах
rng = np.random.default_rng(0)
day_idx = list(res.groupby(res["time"].dt.date).indices.values())
y = res["y"].to_numpy()
arr = {k: res[c].to_numpy() for k, c in cols.items()}
diffs = {"idle-only vs freq_7d": ("LightGBM idle-only", "freq_7d"),
         "all-rows vs freq_7d": ("LightGBM all-rows", "freq_7d")}
for label, (a, b) in diffs.items():
    d = []
    for _ in range(500):
        pick = np.concatenate([day_idx[i] for i in rng.integers(0, len(day_idx), len(day_idx))])
        d.append(average_precision_score(y[pick], arr[a][pick]) -
                 average_precision_score(y[pick], arr[b][pick]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"PR-AUC diff {label}: mean {np.mean(d):+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]")