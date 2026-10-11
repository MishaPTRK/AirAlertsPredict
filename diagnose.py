import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

res = pd.read_parquet("predictions_test.parquet")
res["hybrid"] = res["freq7d"].where(res["now"] == 0, 1.0)  # чесний бейзлайн
cols = {"LightGBM": "p", "freq_7d": "freq7d", "hybrid (now=1 -> 1, else freq_7d)": "hybrid"}


def report(d, title):
    print(f"\n=== {title} | n={len(d)} | base rate={d['y'].mean():.4f}")
    rows = [{"model": n, "PR-AUC": average_precision_score(d["y"], d[c]),
             "ROC-AUC": roc_auc_score(d["y"], d[c]),
             "Brier": brier_score_loss(d["y"], d[c])} for n, c in cols.items()]
    print(pd.DataFrame(rows).round(4).to_string(index=False))


report(res, "all rows")
act = res[res["now"] == 1]
print(f"\nnow==1: n={len(act)} ({len(act)/len(res):.1%} of rows), y rate={act['y'].mean():.4f}, "
      f"share of all positives={act['y'].sum()/res['y'].sum():.1%}")
report(res[res["now"] == 0], "now==0: тривоги немає, чи почнеться/відновиться в наступну годину")

# якщо тривога щойно закінчилась (lag1=1, now=0): типове повторне вмикання
report(res[(res["now"] == 0) & (res["lag1"] == 1)], "now==0 & lag1==1 (щойно відбій)")
report(res[(res["now"] == 0) & (res["lag1"] == 0)], "now==0 & lag1==0 (тихо щонайменше годину)")