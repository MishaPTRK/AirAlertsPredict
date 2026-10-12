import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from src.data import build_panel, load_alerts
from src.features import make_features
from src.model import FEATURES, walk_forward
from src.text_features import TEXT_FEATURES, load_posts, make_text_features

TG_START = pd.Timestamp("2026-08-01", tz="UTC")
TEST_START = pd.Timestamp("2026-09-13", tz="UTC")
TEST_END = pd.Timestamp("2026-10-11", tz="UTC")

panel = build_panel(load_alerts(), "2025-01-01", "2026-10-11")
df = make_features(panel).dropna(subset=["lag24", "freq30d"])
posts = load_posts()
print("posts:", len(posts), "| per day:", round(len(posts) / posts["time"].dt.date.nunique(), 1))
tf = make_text_features(posts, panel["y"].index, list(panel["y"].columns))
df = df.merge(tf, on=["time", "region"], how="left")
df = df[df["time"] >= TG_START + pd.Timedelta(days=2)]  # єдине вікно даних для обох моделей

a = walk_forward(df, TEST_START, TEST_END, train_weeks=5, features=FEATURES)
b = walk_forward(df, TEST_START, TEST_END, train_weeks=5, features=FEATURES + TEXT_FEATURES)

assert not np.allclose(a["p"].to_numpy(), b["p"].to_numpy()), "text features are not being used"

res = a[["time", "region", "y", "freq7d", "p"]].merge(
    b[["time", "region", "p"]], on=["time", "region"], suffixes=("_base", "_text"))

y = res["y"].to_numpy()
rows = [{"model": n, "PR-AUC": average_precision_score(y, res[c]), "Brier": brier_score_loss(y, res[c])}
        for n, c in {"alerts only": "p_base", "alerts + telegram": "p_text", "freq_7d": "freq7d"}.items()]
print("\nn =", len(res), "| base rate =", round(y.mean(), 4))
print(pd.DataFrame(rows).round(4).to_string(index=False))

rng = np.random.default_rng(0)
days = list(res.groupby(res["time"].dt.date).indices.values())
d = []
for _ in range(1000):
    pick = np.concatenate([days[i] for i in rng.integers(0, len(days), len(days))])
    d.append(average_precision_score(y[pick], res["p_text"].to_numpy()[pick])
             - average_precision_score(y[pick], res["p_base"].to_numpy()[pick]))
print(f"PR-AUC diff text vs no text: mean {np.mean(d):+.4f}, "
      f"95% CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")