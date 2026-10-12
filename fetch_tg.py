import sys
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

CHANNEL = "kpszsu"
START = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2026-08-01", tz="UTC")
OUT = "data_raw/tg_kpszsu.csv"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def parse(html):
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for m in soup.select("div.tgme_widget_message[data-post]"):
        t = m.select_one("a.tgme_widget_message_date time")
        if t is None:
            continue
        txt = m.select_one("div.tgme_widget_message_text")
        meta = m.select_one(".tgme_widget_message_meta")
        meta_txt = meta.get_text(" ").lower() if meta else ""
        rows.append({"id": int(m["data-post"].split("/")[1]), "time": t["datetime"],
                     "text": txt.get_text("\n") if txt else "",
                     "edited": ("edited" in meta_txt) or ("змінено" in meta_txt)})
    return rows


def get(url):
    for attempt in range(6):
        try:
            r = requests.get(url, timeout=30, headers=HEADERS)
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.text
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
    return None


def save(rows):
    df = pd.DataFrame(rows).drop_duplicates("id").sort_values("id")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.to_csv(OUT, index=False)
    return df


rows, before = [], None
for page in range(5000):
    html = get(f"https://t.me/s/{CHANNEL}" + (f"?before={before}" if before else ""))
    if html is None:
        print("request failed repeatedly, saving what we have"); break
    batch = parse(html)
    if not batch:
        break
    rows += batch
    new_before = min(b["id"] for b in batch)
    oldest = pd.to_datetime(min(b["time"] for b in batch), utc=True)
    if page % 25 == 0:
        save(rows)
        print(page, "oldest:", oldest, "| total:", len(rows))
    if oldest < START or new_before == before:
        break
    before = new_before
    time.sleep(0.7)

df = save(rows)
print("edited share:", df["edited"].mean().round(3), "| posts:", len(df),
      "| range:", df["time"].min(), "->", df["time"].max())