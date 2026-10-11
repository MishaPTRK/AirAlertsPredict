import numpy as np
import pandas as pd

LONG_H = 24


def load_alerts(path="data_raw/datasets/volunteer_data_en.csv"):
    v = pd.read_csv(path)
    for c in ("started_at", "finished_at"):
        v[c] = pd.to_datetime(v[c], utc=True)
    v = v[v["region"] != "Luhanska oblast"].copy()
    v["hours"] = (v["finished_at"] - v["started_at"]).dt.total_seconds() / 3600
    return v


def build_panel(v, start, end):
    """y[h]: тривога перетинає [h, h+1); now[h]: тривога активна в момент h;
    bad[h]: година покрита тривогою довшою за LONG_H (підозра на пропущене закриття)."""
    hours = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    t0, t1 = hours[0], hours[-1] + pd.Timedelta(hours=1)
    v = v[(v["finished_at"] > t0) & (v["started_at"] < t1)]
    regions = sorted(v["region"].unique())
    arr = {k: np.zeros((len(hours), len(regions)), bool) for k in ("y", "now", "bad")}
    bad_end = np.full((len(hours), len(regions)), np.nan)
    for j, r in enumerate(regions):
        g = v[v["region"] == r]
        for s, f, h in zip(g["started_at"], g["finished_at"], g["hours"]):
            i0 = hours.searchsorted(s.floor("h"))
            i1 = max(hours.searchsorted(f), i0 + 1)
            arr["y"][i0:i1, j] = True
            arr["now"][hours.searchsorted(s):hours.searchsorted(f), j] = True
            if h > LONG_H:
                arr["bad"][i0:i1, j] = True
                bad_end[i0:i1, j] = f.value
    out = {k: pd.DataFrame(a, index=hours, columns=regions) for k, a in arr.items()}
    out["bad_end"] = pd.DataFrame(bad_end, index=hours, columns=regions)
    return out


def make_long(panel):
    """Один рядок = (година T, область). Усі ознаки використовують лише дані до T."""
    y = panel["y"].astype(float)
    hours, regions = y.index, list(y.columns)
    feats = {
        "y": y,
        "now": panel["now"].astype(float),
        "lag1": y.shift(1),
        "lag24": y.shift(24),
        "freq24h": y.shift(1).rolling(24).mean(),
        "freq7d": y.shift(1).rolling(168).mean(),
        "bad": panel["bad"].astype(float),
    }
    df = pd.DataFrame({k: f.to_numpy().ravel() for k, f in feats.items()})
    df["time"] = pd.to_datetime(np.repeat(hours.to_numpy(), len(regions)), utc=True)
    df["region"] = np.tile(regions, len(hours))
    kyiv = hours.repeat(len(regions)).tz_convert("Europe/Kyiv")
    df["how"] = (kyiv.dayofweek * 24 + kyiv.hour).to_numpy()
    df["bad"] = df["bad"].astype(bool)
    df["y"] = df["y"].astype(int)
    return df