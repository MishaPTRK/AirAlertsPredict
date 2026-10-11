import numpy as np
import pandas as pd


def to_hourly(df, region_col, start, end):
    """Бінарна матриця: годину позначено 1, якщо тривога перетинає [h, h+1h). Час у UTC."""
    hours = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    out = {}
    for region, g in df.groupby(region_col):
        active = np.zeros(len(hours), dtype=bool)
        for s, f in zip(g["started_at"], g["finished_at"]):
            i0 = hours.searchsorted(s.floor("h"))
            i1 = max(hours.searchsorted(f, side="left"), i0 + 1)
            active[i0:i1] = True
        out[region] = active
    return pd.DataFrame(out, index=hours)