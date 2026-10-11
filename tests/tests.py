import numpy as np
import pandas as pd

from src.data import build_panel
from src.features import make_features
from src.model import FEATURES, split_week

START, END = "2026-01-01", "2026-03-17"


def synthetic_alerts(seed=0):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(START, tz="UTC")
    rows = []
    for region in ["A", "B", "C"]:
        t = t0
        while t < t0 + pd.Timedelta(days=75):
            t += pd.Timedelta(minutes=int(rng.integers(30, 1200)))
            rows.append((region, t, t + pd.Timedelta(minutes=int(rng.integers(5, 240)))))
    rows.append(("A", t0 + pd.Timedelta(days=45), t0 + pd.Timedelta(days=45, hours=30)))  # довга
    v = pd.DataFrame(rows, columns=["region", "started_at", "finished_at"])
    v["hours"] = (v["finished_at"] - v["started_at"]).dt.total_seconds() / 3600
    return v


def truncate(v, T):
    """Те, що було відоме в момент T: нічого, що почалось після T; тривога, що триває, лишається активною."""
    v = v[v["started_at"] <= T].copy()
    v["finished_at"] = v["finished_at"].where(v["finished_at"] <= T, T + pd.Timedelta(seconds=1))
    v["hours"] = (v["finished_at"] - v["started_at"]).dt.total_seconds() / 3600
    return v


def feature_frame(v):
    return make_features(build_panel(v, START, END))


def test_features_do_not_depend_on_future():
    v = synthetic_alerts()
    T = pd.Timestamp("2026-02-20", tz="UTC")
    cols = ["time", "region"] + [c for c in FEATURES if c != "region_id"]
    full = feature_frame(v)
    cut = feature_frame(truncate(v, T))
    a = full[full["time"] <= T][cols].sort_values(["time", "region"]).reset_index(drop=True)
    b = cut[cut["time"] <= T][cols].sort_values(["time", "region"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_detector_catches_a_deliberately_leaky_feature():
    """Контрольний тест: ознака з майбутнього має бути помічена, інакше тест вище нічого не вартий."""
    v = synthetic_alerts()
    T = pd.Timestamp("2026-02-20", tz="UTC")
    leaky_full = build_panel(v, START, END)["y"].shift(-48)
    leaky_cut = build_panel(truncate(v, T), START, END)["y"].shift(-48)
    idx = leaky_full.index[leaky_full.index <= T]
    assert not leaky_full.loc[idx].fillna(-1).equals(leaky_cut.loc[idx].fillna(-1))


def test_walk_forward_split_uses_only_the_past():
    df = feature_frame(synthetic_alerts()).dropna(subset=["lag24", "freq30d"])
    ws = pd.Timestamp("2026-02-18", tz="UTC")
    tr, te = split_week(df, ws, train_weeks=4)
    assert tr["time"].max() < ws <= te["time"].min()


def test_long_alert_is_masked_in_train_only_once_it_is_closed():
    df = feature_frame(synthetic_alerts()).dropna(subset=["lag24", "freq30d"])
    ongoing = pd.Timestamp("2026-02-16", tz="UTC")  # тривога триває до 16.02 06:00
    closed = pd.Timestamp("2026-02-18", tz="UTC")
    assert split_week(df, ongoing, 4)[0]["bad"].any()
    assert not split_week(df, closed, 4)[0]["bad"].any()