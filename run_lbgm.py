import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.data import load_alerts, build_panel
from src.features import make_features

TEST_START = pd.Timestamp("2026-09-13", tz="UTC")
TEST_END = pd.Timestamp("2026-10-11", tz="UTC")
TRAIN_WEEKS = 26
FEATURES = ["now", "lag1", "lag2", "lag3", "lag24", "freq3h", "freq6h", "freq24h",
            "freq7d", "freq30d", "since_last", "others_now", "others_prev",
            "hour", "dow", "region_id"]

v = load_alerts()
df = make_features(build_panel(v, "2025-01-01", "2026-10-11"))
df = df.dropna(subset=["lag24", "freq30d"])
df = df[~df["bad"]]

parts = []
for ws in pd.date_range(TEST_START, TEST_END, freq="7D", inclusive="left"):
    we = ws + pd.Timedelta(days=7)
    train = df[(df["time"] < ws) & (df["time"] >= ws - pd.Timedelta(weeks=TRAIN_WEEKS))]
    test = df[(df["time"] >= ws) & (df["time"] < we)].copy()
    model = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                           min_child_samples=50, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, random_state=0, verbose=-1)
    model.fit(train[FEATURES], train["y"])
    test["p"] = model.predict_proba(test[FEATURES])[:, 1]
    parts.append(test)
    print(ws.date(), "train rows:", len(train), "| PR-AUC:",
          round(average_precision_score(test["y"], test["p"]), 4))

res = pd.concat(parts)
cols = {"LightGBM": "p", "freq_7d": "freq7d", "active_now": "now"}
rows = [{"model": n, "PR-AUC": average_precision_score(res["y"], res[c]),
         "ROC-AUC": roc_auc_score(res["y"], res[c]),
         "Brier": brier_score_loss(res["y"], res[c])} for n, c in cols.items()]
print("\ntest rows:", len(res), "| base rate:", round(res["y"].mean(), 4))
print(pd.DataFrame(rows).round(4).to_string(index=False))

res["bin"] = pd.cut(res["p"], [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0], include_lowest=True)
print("\ncalibration:\n", res.groupby("bin", observed=True).agg(
    n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean")).round(3).to_string())

per = res.groupby("region").apply(lambda g: pd.Series({
    "base": g["y"].mean(),
    "PR-AUC lgbm": average_precision_score(g["y"], g["p"]) if g["y"].nunique() > 1 else float("nan"),
    "PR-AUC freq7d": average_precision_score(g["y"], g["freq7d"]) if g["y"].nunique() > 1 else float("nan"),
}), include_groups=False)
print("\nper region:\n", per.round(3).to_string())
res.drop(columns="bin").to_parquet("predictions_test.parquet")