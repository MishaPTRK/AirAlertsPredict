import re

import numpy as np
import pandas as pd

REGION_PATTERNS = {
    "Cherkaska oblast": r"черкас",
    "Chernihivska oblast": r"чернігів|чернігів",
    "Chernivetska oblast": r"чернівц|буковин",
    "Dnipropetrovska oblast": r"дніпро|дніпропетров|павлоград|кам.янськ",
    "Donetska oblast": r"донеччин|донецьк|краматорськ|слов.янськ|покровськ",
    "Ivano-Frankivska oblast": r"франківськ|прикарпат",
    "Kharkivska oblast": r"харків|харківщин",
    "Khersonska oblast": r"херсон",
    "Khmelnytska oblast": r"хмельниц",
    "Kirovohradska oblast": r"кіровоград|кропивниц",
    "Kyiv City": r"київ(?!щ|ськ)|києв",
    "Kyivska oblast": r"київщ|київськ",
    "Lvivska oblast": r"львів|львівщ",
    "Mykolaivska oblast": r"миколаїв",
    "Odeska oblast": r"одес|одещ",
    "Poltavska oblast": r"полтав",
    "Rivnenska oblast": r"рівн(?!ин)",
    "Sumska oblast": r"\bсум(?:и|ах|щ|ськ)",
    "Ternopilska oblast": r"терноп",
    "Vinnytska oblast": r"вінниц|вінниччин",
    "Volynska oblast": r"волин|луцьк",
    "Zakarpatska oblast": r"закарпат|ужгород",
    "Zaporizka oblast": r"запоріж",
    "Zhytomyrska oblast": r"житомир",
}
THREAT = r"бпла|шахед|ракет|крилат|балістич|курсом на"
TEXT_FEATURES = ["tg_reg_1h", "tg_reg_3h", "tg_reg_24h", "tg_nat_1h", "tg_nat_threat_3h"]


def load_posts(path="data_raw/tg_kpszsu.csv", drop_edited=True):
    p = pd.read_csv(path)
    p["time"] = pd.to_datetime(p["time"], utc=True)
    p["edited"] = p["edited"].astype(bool)
    if drop_edited:  # збережений текст міг бути змінений вже після моменту прогнозу
        p = p[~p["edited"]]
    return p[["time", "text"]]


def make_text_features(posts, hours, regions):
    """Ознака в момент T використовує лише пости, опубліковані строго до T."""
    posts = posts[(posts["time"] >= hours[0]) & (posts["time"] < hours[-1] + pd.Timedelta(hours=1))]
    text = posts["text"].fillna("").str.lower()
    bins = posts["time"].dt.floor("h")

    def hourly(mask):
        s = posts[mask.to_numpy()].groupby(bins[mask.to_numpy()]).size()
        return s.reindex(hours, fill_value=0).astype(float)

    reg = pd.DataFrame({r: hourly(text.str.contains(REGION_PATTERNS[r], regex=True)).to_numpy()
                        for r in regions}, index=hours)
    nat_all = hourly(pd.Series(True, index=text.index))
    nat_threat = hourly(text.str.contains(THREAT, regex=True))

    n = len(regions)
    out = {
        "tg_reg_1h": reg.shift(1),
        "tg_reg_3h": reg.rolling(3).sum().shift(1),
        "tg_reg_24h": reg.rolling(24).sum().shift(1),
    }
    df = pd.DataFrame({k: v.to_numpy().ravel() for k, v in out.items()})
    df["tg_nat_1h"] = np.repeat(nat_all.shift(1).to_numpy(), n)
    df["tg_nat_threat_3h"] = np.repeat(nat_threat.rolling(3).sum().shift(1).to_numpy(), n)
    df["time"] = pd.to_datetime(np.repeat(hours.to_numpy(), n), utc=True)
    df["region"] = np.tile(regions, len(hours))
    return df