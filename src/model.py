import pandas as pd
from lightgbm import LGBMClassifier

FEATURES = ["now", "lag1", "lag2", "lag3", "lag24", "freq3h", "freq6h", "freq24h",
            "freq7d", "freq30d", "since_last", "others_now", "others_prev",
            "hour", "dow", "region_id"]
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)


def split_week(df, ws, train_weeks=26):
    """Навчання: лише рядки до ws. Довгу тривогу маскуємо, тільки якщо вона вже закрита до ws.
    Тест: тиждень з ws; мітки довгих тривог вважаємо помилками даних і виключаємо."""
    tr = df[(df["time"] >= ws - pd.Timedelta(weeks=train_weeks)) & (df["time"] < ws)]
    tr = tr[~(tr["bad"] & (tr["bad_end"] <= ws))]
    te = df[(df["time"] >= ws) & (df["time"] < ws + pd.Timedelta(days=7))]
    return tr, te[~te["bad"]]


def walk_forward(df, test_start, test_end, train_weeks=26, idle_only=True):
    if idle_only:
        df = df[df["now"] == 0]
    parts = []
    for ws in pd.date_range(test_start, test_end, freq="7D", inclusive="left"):
        tr, te = split_week(df, ws, train_weeks)
        model = LGBMClassifier(**PARAMS).fit(tr[FEATURES], tr["y"])
        te = te.copy()
        te["p"] = model.predict_proba(te[FEATURES])[:, 1]
        parts.append(te)
    return pd.concat(parts)