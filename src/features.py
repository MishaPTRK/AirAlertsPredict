import numpy as np
import pandas as pd


def make_features(panel):
    y = panel["y"].astype(float)
    now = panel["now"].astype(float)
    hours, regions = y.index, list(y.columns)
    prev = y.shift(1)

    # скільки годин минуло від останньої активної години (до T), обрізано до 168
    pos = pd.DataFrame(np.repeat(np.arange(len(hours))[:, None], len(regions), axis=1),
                       index=hours, columns=regions)
    since = (pos - pos.where(prev == 1).ffill()).fillna(168).clip(upper=168)

    # скільки ІНШИХ областей активні в момент T і були активні годину тому
    oth_now = pd.DataFrame(now.sum(axis=1).to_numpy()[:, None] - now.to_numpy(),
                           index=hours, columns=regions)
    oth_prev = pd.DataFrame(prev.sum(axis=1).to_numpy()[:, None] - prev.to_numpy(),
                            index=hours, columns=regions)

    feats = {
        "bad_end": panel["bad_end"],
        "y": y, "now": now, "bad": panel["bad"].astype(float),
        "lag1": prev, "lag2": y.shift(2), "lag3": y.shift(3), "lag24": y.shift(24),
        "freq3h": prev.rolling(3).mean(), "freq6h": prev.rolling(6).mean(),
        "freq24h": prev.rolling(24).mean(), "freq7d": prev.rolling(168).mean(),
        "freq30d": prev.rolling(720).mean(),
        "since_last": since, "others_now": oth_now, "others_prev": oth_prev,
    }
    df = pd.DataFrame({k: f.to_numpy().ravel() for k, f in feats.items()})
    df["time"] = pd.to_datetime(np.repeat(hours.to_numpy(), len(regions)), utc=True)
    df["region"] = np.tile(regions, len(hours))
    kyiv = hours.repeat(len(regions)).tz_convert("Europe/Kyiv")
    df["hour"] = kyiv.hour.to_numpy()
    df["dow"] = kyiv.dayofweek.to_numpy()
    df["bad"] = df["bad"].astype(bool)
    df["bad_end"] = pd.to_datetime(df["bad_end"], unit="ns", utc=True)
    df["y"] = df["y"].astype(int)
    df["region_id"] = df["region"].astype("category")
    return df