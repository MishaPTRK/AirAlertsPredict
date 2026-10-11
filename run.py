import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from src.data import build_panel, load_alerts
from src.features import make_features
from src.model import walk_forward

TEST_START = pd.Timestamp("2026-09-13", tz="UTC")
TEST_END = pd.Timestamp("2026-10-11", tz="UTC")  # виключно; 11.10 неповна доба

df = make_features(build_panel(load_alerts(), "2025-01-01", "2026-10-11"))
df = df.dropna(subset=["lag24", "freq30d"])
res = walk_forward(df, TEST_START, TEST_END)

base = df[(df["now"] == 0) & (df["time"] < TEST_START) & ~df["bad"]]["y"].mean()
res["const"] = base
cols = {"LightGBM": "p", "freq_7d": "freq7d", "freq_24h": "freq24h", "constant": "const"}
y = res["y"].to_numpy()
table = {n: {"PR-AUC": average_precision_score(y, res[c]), "Brier": brier_score_loss(y, res[c])}
         for n, c in cols.items()}

rng = np.random.default_rng(0)
days = list(res.groupby(res["time"].dt.date).indices.values())
diffs = []
for _ in range(1000):
    pick = np.concatenate([days[i] for i in rng.integers(0, len(days), len(days))])
    diffs.append(average_precision_score(y[pick], res["p"].to_numpy()[pick])
                 - average_precision_score(y[pick], res["freq7d"].to_numpy()[pick]))
ci = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]

out = {"test_rows": len(res), "base_rate": float(y.mean()), "metrics": table, "pr_auc_diff_ci95": ci}
print(json.dumps(out, indent=2, ensure_ascii=False))
json.dump(out, open("results.json", "w"), indent=2, ensure_ascii=False)