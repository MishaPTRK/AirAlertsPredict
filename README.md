# Next-hour air-raid alert forecasting for Ukrainian oblasts

Stage 2 project: time-series analysis of air-raid alerts in Ukraine.
At the start of every UTC hour **T**, forecast for each of 24 oblasts whether an alert will be active at
any moment in **[T, T+1h)**, using only information available at T.

The work was done in one AI-assisted session (the full chat log is submitted separately). Every number in
this README was copied from an actual run of the code in this repository during that session.

## Summary

- **Main part (alert history only).** Weekly-retrained LightGBM, test on the 4 latest weeks
  (2026-09-13 … 2026-10-10). On hours where no alert is active at T it reaches PR-AUC **0.513** and Brier
  **0.152**, against 0.492 / 0.161 for the best simple baseline (alert share over the last 7 days). The
  PR-AUC gain is **+0.02**, bootstrap 95% CI **[+0.010, +0.032]**: real in this window, but modest.
- **The first version looked much better and was misleading.** Over all rows LightGBM scored PR-AUC 0.80,
  but a trivial rule ("if an alert is active now, predict 1; otherwise use the 7-day share") scores 0.79.
  So the evaluation was redefined to the hours where the question is actually open.
- **Additional part (text, optional).** Added features from the official Air Force Telegram channel
  (14,558 posts collected). Result: PR-AUC +0.009, 95% CI [-0.001, +0.019], so **inconclusive**.
- **Data caveats found on the way:** the official feed in the source repository stops on 2026-09-07 and
  became district-level in Dec 2025, so the volunteer feed is used as main data; the two feeds disagree
  strongly in several oblasts; a few alerts are obviously missing their end message.

---

# Part 1. Main task: alert history

## 1.1 Task definition

- One row = (hour T in UTC, oblast). Target `y` = 1 if an alert intersects [T, T+1h).
- Features use only hours before T plus the alert state at the moment T (`now`, known in real time).
- Time is handled in UTC. Kyiv-time hour and weekday are used only as features.
- **Main evaluation is on rows with no active alert at T (`now == 0`):** "will an alert start or restart in
  the next hour?". If an alert is already active at T, the target is 1 by definition, so those rows are not
  forecasts (see 1.5).
- Test period: **2026-09-13 00:00 UTC … 2026-10-11 00:00 UTC** (4 full weeks). 2026-10-11 is excluded
  because the data for that day is incomplete.

## 1.2 Data

### Sources inspected

Repository: [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset)
(cloned with `--depth 1` on 2026-10-11; the last commit was "Add new data", 2026-10-11 04:50 UTC).

| File | Rows | Columns | Time range (started_at) | Notes |
|---|---|---|---|---|
| `official_data_en.csv` | 299,275 | oblast, raion, hromada, level, started_at, finished_at, source | 2022-03-15 16:10 → **2026-09-07 05:28 UTC** | levels: raion 132,251, oblast 130,235, hromada 36,789; 25 oblasts; no nulls in time columns; all timestamps `+00:00` |
| `volunteer_data_en.csv` | 115,989 | region, started_at, finished_at, naive | 2022-02-25 16:36 → **2026-10-11 04:36 UTC** | oblast level only; 25 regions |

Findings that drove the decisions:

- The repository README says both datasets are "updated daily", but the official file ends on 2026-09-07
  (its oblast-level rows end on 2026-09-05). The commit date only reflects the automatic update job, not new
  rows. The repository README also states that since December 2025 official alerts are mostly at raion
  (district) level, so oblast-level rows are almost absent in the recent official data.
- The volunteer feed (from the eTryvoga channel) is the only fresh source at oblast level, with one
  consistent method since 2022. It is therefore used as **main data**. The official feed is used for
  cross-checks only.
- The repository README states that two permanent sirens (Luhansk oblast and Crimea) are not listed in the
  datasets. Luhansk appears in the files, but never as an active alert in the hours checked (share 0.000
  in both feeds), so it is excluded as a constant. Crimea is not in the data. **24 oblasts remain.**

### Quality of the volunteer feed

- `naive = True` (no end message, `finished_at = started_at + 30 min`): **5.0%** of alerts overall. Share by
  year: 2022 2.0%, 2023 6.1%, 2024 5.6%, 2025 5.8%, 2026 4.7%. Stable since 2023, so these alerts are kept.
- No negative durations. Duration in minutes: mean 47.8, median 30, 90th percentile 103.9,
  99th percentile 266.2, maximum 10,695.9 (about 7.4 days).
- **Alerts longer than 24 h: 14** (longer than 12 h: 28, longer than 48 h: 10). By oblast: Chernivtsi 3,
  Ternopil 3, Khmelnytskyi 2, Zhytomyr 2, Zakarpattia 2, Lviv 1, Zaporizhzhia 1. The longest:

| Oblast | Start (UTC) | Hours |
|---|---|---|
| Zaporizka | 2022-03-04 08:02 | 178.3 |
| Chernivetska | 2026-04-17 16:55 | 177.2 |
| Zakarpatska | 2026-04-17 16:52 | 122.4 |
| Ternopilska | 2026-04-17 16:55 | 121.0 |
| Khmelnytska | 2026-04-17 16:55 | 108.0 |
| Ternopilska | 2026-08-28 02:14 | 93.9 |
| Lvivska | 2026-01-29 01:30 | 90.7 |
| Ternopilska | 2026-09-18 17:11 | 79.0 |
| Chernivetska | 2025-03-03 09:45 | 77.9 |
| Khmelnytska | 2026-08-04 17:23 | 69.9 |

  Four western oblasts start an alert within 3 minutes on 2026-04-17 and end days later, which looks like
  missing end messages rather than real alerts. The choice of the 24 h threshold hardly matters: 10 of the
  14 alerts exceed 48 h. Hours covered by such alerts are flagged `bad` (0.3% of rows).

### The two feeds disagree

Event counts (Jun–Nov 2025, oblast level) differ strongly: the ratio volunteer/official ranges from 0.53
(Kyiv City, 280 vs 526) to 2.47 (Dnipropetrovsk, 1,607 vs 650). More informative is the hourly activity,
compared for 2025-06-01 … 2025-12-01 (share of hours with an alert; Jaccard overlap of active hours):

| Oblast | Volunteer share | Official share | Jaccard |
|---|---|---|---|
| Cherkaska | 0.162 | 0.083 | 0.478 |
| Chernihivska | 0.344 | 0.207 | 0.306 |
| Chernivetska | 0.051 | 0.036 | 0.620 |
| Dnipropetrovska | 0.455 | 0.134 | 0.205 |
| Donetska | 0.488 | 0.759 | 0.540 |
| Ivano-Frankivska | 0.037 | 0.036 | 0.706 |
| Kharkivska | 0.488 | 0.261 | 0.287 |
| Khersonska | 0.265 | 0.167 | 0.427 |
| Khmelnytska | 0.044 | 0.029 | 0.414 |
| Kirovohradska | 0.220 | 0.122 | 0.412 |
| Kyiv City | 0.107 | 0.142 | 0.701 |
| Kyivska | 0.162 | 0.092 | 0.311 |
| Lvivska | 0.085 | 0.038 | 0.337 |
| Mykolaivska | 0.259 | 0.185 | 0.473 |
| Odeska | 0.166 | 0.146 | 0.681 |
| Poltavska | 0.290 | 0.147 | 0.307 |
| Rivnenska | 0.050 | 0.041 | 0.556 |
| Sumska | 0.466 | 0.302 | 0.293 |
| Ternopilska | 0.041 | 0.042 | 0.676 |
| Vinnytska | 0.071 | 0.060 | 0.603 |
| Volynska | 0.183 | 0.046 | 0.178 |
| Zakarpatska | 0.046 | 0.031 | 0.628 |
| Zaporizka | 0.417 | 0.451 | 0.674 |
| Zhytomyrska | 0.082 | 0.101 | 0.608 |

Quiet and western oblasts and Kyiv City agree reasonably (Jaccard 0.6-0.7); front-line and some others
(Dnipropetrovsk, Kharkiv, Sumy, Poltava, Kyivska oblast, Volyn) differ by a factor of 1.5-3 or more. Donetsk
goes the other way (official 0.759 vs volunteer 0.488). The cause is unknown (different aggregation,
segmentation, or alert types are possible explanations, none verified). The target of this project is
therefore **"alert active in the volunteer feed"**, not "official alert".

### Western oblasts: Lviv and Volyn

Monthly share of active hours (volunteer vs official) for selected months:

| Month | Lviv vol. | Lviv off. | Volyn vol. | Volyn off. |
|---|---|---|---|---|
| 2025-09 | 0.022 | 0.036 | 0.367 | 0.049 |
| 2025-10 | 0.030 | 0.038 | 0.333 | 0.038 |
| 2025-11 | **0.314** | 0.040 | 0.214 | 0.032 |
| 2026-07 | 0.202 | 0.005 | 0.355 | 0.005 |
| 2026-08 | 0.281 | 0.004 | 0.284 | 0.008 |

Lviv volunteer alerts per month: 17 in 2025-10, **115 in 2025-11**, then 44-141 per month (141 in 2026-08).
Median duration changed only from 0.41 to 0.67 h, so it is the number of alerts that jumped, not their
length. Neighbouring oblasts (Ivano-Frankivsk, Rivne, Ternopil) did not jump in the same month. Official
data for the west is close to zero in summer 2026. Whether this is a real change in activity, a different
way of recording, or a gap in the official feed could not be established. Volyn shows a persistent 5-9x
gap between the feeds for a year. September 2026 in this table is partial (it ends on 2026-09-06).

### Non-stationarity

Mean share of active hours across oblasts (volunteer feed, excluding Luhansk):

| Month | 2025-01 | 2025-06 | 2025-09 | 2025-12 | 2026-03 | 2026-06 | 2026-08 | 2026-09 | 2026-10 (partial) |
|---|---|---|---|---|---|---|---|---|---|
| Share | 0.215 | 0.191 | 0.210 | 0.215 | 0.256 | 0.253 | 0.287 | 0.326 | 0.339 |

Kharkiv rises from 0.402 (2025-01) to 0.701 (2026-09), Dnipropetrovsk from 0.477 to 0.750, Kyiv City from
0.148 to 0.285. There is no abrupt break around December 2025 (when the official feed changed to districts),
but there is a steady increase. This is not a problem for training by itself (the trend is real), but it
means that models trained on old data under-predict in the newest period, so the model is retrained weekly
on a recent window and calibration is checked.

## 1.3 Cleaning rules

- Luhansk is excluded. `naive` alerts are kept.
- Alerts longer than 24 h mark their hours as `bad`. Those hours are excluded from every test set.
  In training, they are excluded only once the alert has been closed before the forecast week starts
  (alert length is only known after it ends; using it earlier would use information from the future).
  The final version of the mask was written after noticing that the first version did not respect this.
- Features never use the `bad` flag.

## 1.4 Model

- **Features** (all computed from data before T, plus the alert state at T): `now`; lags 1, 2, 3, 24 h;
  alert frequency over 3 h, 6 h, 24 h, 7 d, 30 d; hours since the last active hour (capped at 168); number
  of other oblasts active now and one hour ago; Kyiv hour and weekday; region.
- **LightGBM**: 300 trees, learning rate 0.05, 31 leaves, `min_child_samples=50`, subsample 0.8,
  colsample 0.8, seed 0.
- **Walk-forward**: retrained every 7 days (week starts 2026-09-13, 09-20, 09-27, 10-04) on the previous
  26 weeks. Trained on `now == 0` rows only (the same score as training on all rows: 0.5127 vs 0.5123).
- **Baselines**: persistence (alert in the previous hour), `active_now`, alert share over the last 24 h and
  7 d, mean for the same hour of week over the last 26 weeks, constant (training base rate), and the
  hybrid rule "active now -> 1, otherwise 7-day share".

## 1.5 Results

### Baselines on all rows (before the diagnostic)

Test rows 16,048 (16,128 without the long-alert mask). Test base rate 0.332 vs training base rate 0.267.

| Baseline | PR-AUC | Brier |
|---|---|---|
| persistence (lag 1) | 0.4986 | 0.2615 |
| active_now | 0.5726 | 0.2127 |
| freq_24h | 0.5662 | 0.1863 |
| freq_7d | 0.5783 | 0.1842 |
| hour_of_week_mean | 0.5430 | 0.2000 |
| constant | 0.3323 | 0.2262 |

Sensitivity to cleaning (PR-AUC of `freq_7d` / `active_now`): masked 0.578 / 0.573; not masked
0.578 / 0.581; masked and without Volyn and Lviv (14,704 rows) 0.591 / 0.573. Conclusions do not change.

### First LightGBM, all rows (why 0.80 is misleading)

| Model | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|
| LightGBM | 0.8004 | 0.8515 | 0.1334 |
| freq_7d | 0.5783 | 0.7452 | 0.1842 |
| active_now | 0.5726 | 0.6799 | 0.2127 |
| hybrid (active now -> 1, else freq_7d) | 0.7909 | 0.8420 | 0.1414 |

Weekly PR-AUC was stable (0.792, 0.806, 0.801, 0.804). However, rows where an alert is already active
(`now == 1`) are **1,919 rows = 12.0% of the test set and 36.0% of all positives, with y = 1.000**. They
are not forecasts. The trivial hybrid rule is only 0.009 PR-AUC below LightGBM.

### Honest evaluation: rows with no active alert (n = 14,129, base rate 0.2416)

Final result from `run.py`:

| Model | PR-AUC | Brier |
|---|---|---|
| **LightGBM (weekly walk-forward)** | **0.5128** | **0.1515** |
| freq_7d | 0.4925 | 0.1606 |
| freq_24h | 0.4635 | 0.1676 |
| constant (training base rate) | 0.2416 | 0.1935 |

PR-AUC difference LightGBM vs `freq_7d`: **+0.021, bootstrap 95% CI [+0.010, +0.032]** (1,000 resamples
by day). Training only on `now == 0` rows gives the same quality as training on all rows
(PR-AUC 0.5127 vs 0.5123, Brier 0.1515 vs 0.1516; difference vs `freq_7d` +0.0208 [+0.0099, +0.0322] and
+0.0206 [+0.0085, +0.0324]).

Subsets of the no-active-alert rows (all-rows model):

| Subset | n | base rate | LightGBM PR-AUC / Brier | freq_7d PR-AUC / Brier |
|---|---|---|---|---|
| all rows with no active alert | 14,129 | 0.242 | 0.5123 / 0.1516 | 0.4925 / 0.1606 |
| alert ended in the previous hour (`lag1 == 1`) | 3,410 | 0.386 | 0.6256 / 0.1989 | 0.6076 / 0.2063 |
| quiet for at least an hour (`lag1 == 0`) | 10,719 | 0.196 | 0.4222 / 0.1365 | 0.3965 / 0.1461 |

The model adds a moderate amount over frequency-based baselines; it does not solve the problem. The
bootstrap treats days as independent although oblasts within one day are correlated, and the window is only
28 days, so the interval should be read as indicative.

### Per-oblast PR-AUC (all-rows model, test period)

These numbers include the trivially active rows and are shown only to see where the signal comes from.

| Oblast | Base rate | LightGBM | freq_7d |
|---|---|---|---|
| Cherkaska | 0.350 | 0.732 | 0.357 |
| Chernihivska | 0.372 | 0.742 | 0.379 |
| Chernivetska | 0.068 | 0.528 | 0.073 |
| Dnipropetrovska | 0.737 | 0.875 | 0.730 |
| Donetska | 0.286 | 0.782 | 0.260 |
| Ivano-Frankivska | 0.018 | 0.472 | 0.022 |
| Kharkivska | 0.741 | 0.898 | 0.762 |
| Khersonska | 0.249 | 0.653 | 0.243 |
| Khmelnytska | 0.131 | 0.546 | 0.193 |
| Kirovohradska | 0.440 | 0.743 | 0.443 |
| Kyiv City | 0.286 | 0.622 | 0.270 |
| Kyivska | 0.585 | 0.786 | 0.588 |
| Lvivska | 0.220 | 0.577 | 0.217 |
| Mykolaivska | 0.638 | 0.826 | 0.652 |
| Odeska | 0.408 | 0.670 | 0.409 |
| Poltavska | 0.438 | 0.731 | 0.432 |
| Rivnenska | 0.195 | 0.602 | 0.230 |
| Sumska | 0.469 | 0.789 | 0.484 |
| Ternopilska | 0.061 | 0.610 | 0.084 |
| Vinnytska | 0.235 | 0.564 | 0.240 |
| Volynska | 0.314 | 0.751 | 0.302 |
| Zakarpatska | 0.000 | n/a | n/a |
| Zaporizka | 0.379 | 0.794 | 0.356 |
| Zhytomyrska | 0.324 | 0.651 | 0.310 |

Zakarpattia has no positive rows in the test period, so its metrics are undefined.

### Calibration (all-rows model)

| Predicted bin | n | mean predicted | observed |
|---|---|---|---|
| 0-0.05 | 2,963 | 0.015 | 0.026 |
| 0.05-0.1 | 1,051 | 0.075 | 0.126 |
| 0.1-0.2 | 3,424 | 0.151 | 0.183 |
| 0.2-0.3 | 2,609 | 0.247 | 0.253 |
| 0.3-0.4 | 1,814 | 0.345 | 0.342 |
| 0.4-0.5 | 844 | 0.446 | 0.491 |
| 0.5-0.6 | 672 | 0.546 | 0.558 |
| 0.6-0.7 | 435 | 0.647 | 0.644 |
| 0.7-0.8 | 257 | 0.742 | 0.724 |
| 0.8-0.9 | 60 | 0.823 | 0.750 |
| 0.9-1.0 | 1,919 | 0.999 | 1.000 |

Calibration is good in the middle and slightly under-confident at low probabilities (consistent with the
upward trend in alert share). The last bin is the trivially active rows. The calibration of the final
no-active-alert model was not re-plotted separately.

## 1.6 Leakage prevention and tests

Design rules:

1. Every feature at time T is built from hours before T (`shift`/`rolling` on shifted series) plus the alert
   state at T, which is known in real time.
2. Weekly training uses only rows strictly before the forecast week; nothing (scaler, encoder, statistics)
   is fitted on test rows.
3. The long-alert mask uses an alert's length only after it is closed (see 1.3).

Tests (`python -m pytest tests -q`, 4 passed in 1.65 s on synthetic data):

- `test_features_do_not_depend_on_future`: data after T are removed; features of all rows at times <= T
  must be identical.
- `test_detector_catches_a_deliberately_leaky_feature`: control test with a deliberately leaky feature
  (`shift(-48)`); the check must fail on it, which shows that the first test can detect leakage.
- `test_walk_forward_split_uses_only_the_past`: train maximum time < forecast week start <= test minimum time.
- `test_long_alert_is_masked_in_train_only_once_it_is_closed`: masking respects the closing time.

The tests check the feature and split logic on synthetic data; they do not replace running the pipeline on
real data.

---

# Part 2. Additional task: text data (official Air Force Telegram channel)

## 2.1 Data collection

- Source: the official Air Force channel (`kpszsu`), public web preview `t.me/s/kpszsu`, collected by
  `fetch_tg.py` (no account needed; 0.7 s pause between requests; retries on rate limits).
- Collected **14,558 posts** from 2026-07-31 23:03 to 2026-10-12 11:51 UTC (about 200 per day, 725 pages).
  **2.1%** of the posts were marked as edited and were dropped (the stored text may differ from the one
  published at the time), leaving **14,251** posts used for features (192.6 per day).
- Only the publication time and text are used.

## 2.2 Features (`src/text_features.py`)

Computed by simple keyword matching on the lower-cased text, with Ukrainian stems for each oblast
(including major cities):

| Feature | Meaning (all strictly from posts published before T) |
|---|---|
| `tg_reg_1h` | posts mentioning the oblast in the last hour |
| `tg_reg_3h` | ... in the last 3 hours |
| `tg_reg_24h` | ... in the last 24 hours |
| `tg_nat_1h` | all posts in the last hour |
| `tg_nat_threat_3h` | posts about drones or missiles in the last 3 hours |

## 2.3 Experiment and result

Same LightGBM, same four test weeks, same rows (`now == 0`, n = 14,129), with and without the text features.
Because the collection goes back only to August, **both models are trained on a 5-week window**, so these
numbers are not comparable with the main table (26 weeks).

| Model | PR-AUC | Brier |
|---|---|---|
| alerts only (5-week training) | 0.4927 | 0.1551 |
| alerts + Telegram | 0.5020 | 0.1533 |
| freq_7d | 0.4925 | 0.1606 |

PR-AUC difference with vs without text: **+0.009, bootstrap 95% CI [-0.0007, +0.0187]**. The interval
contains zero, so the improvement is **not established**; Brier moved from 0.1551 to 0.1533. The experiment
was run once, with no tuning on the test weeks. A second finding: with only 5 weeks of training, the
alerts-only model is no better than `freq_7d` (0.4927 vs 0.4925), whereas with 26 weeks it is (0.5128).

## 2.4 Leakage handling and caveats for the text part

- A post is used only if its publication time is strictly before T; features at T are counted from
  hourly bins shifted by one hour.
- Edited posts are dropped; deleted posts are invisible, so the sample is slightly biased. The collection was
  done on 2026-10-12, so the channel could have changed old posts silently; this risk cannot be measured.
- No separate automated leakage test for the text features had been run at the time of writing; the guarantees
  above come from the construction of the features.
- Keyword matching misses district names, unusual spellings and wording like "in the south". The share of posts
  that mention an oblast was not measured, and the importance of the text features was not analysed.
- The channel is an official source, but its publication practice may change; the model has no way to adapt.

---

# Limitations

- The target is the volunteer feed. The official feed differs strongly in several oblasts; which one is closer
  to reality could not be established. Official data ends on 2026-09-07 and is district-level since Dec 2025.
- Lviv and Volyn have large unexplained differences between the feeds. Excluding them changes baseline
  scores by about 0.01 PR-AUC.
- Only 4 test weeks. An hour counts as active even if an alert lasted a few minutes. The bootstrap treats
  days as independent. Alert share rises over time, so models under-predict slightly at low probabilities.
- The 24 h threshold for suspect alerts is a judgement call; the mask touches 0.3% of rows.
- The text experiment is short (5 weeks of training) and inconclusive.

# Process notes: what went wrong and how it was fixed

1. **First model: PR-AUC 0.80.** A diagnostic showed that 36% of positives were trivially known from the
   current alert state and a trivial hybrid rule scored 0.79. Evaluation was moved to rows with no active
   alert and the hybrid baseline was added; a bootstrap CI was added to quantify the real gain.
2. **Dataset freshness.** The first assumption (updated daily, official feed usable) was wrong: the official
   file ended 5 weeks earlier, and every file in the repository was checked before choosing the volunteer feed.
3. **Wrong assumptions in code**, found by running it: the volunteer file uses `region`, not `oblast`;
   oblast-level official rows disappear after Dec 2025.
4. **Hypotheses revised by data.** The Lviv jump was first suspected to be missing end messages; the data
   showed that the number of alerts changed, not their duration, and could not decide between real activity and
   a recording change. The "growing share" was first treated as a problem and then understood as a trend
   to handle by weekly retraining.
5. **Mask definition.** The first long-alert mask used the alert's final length at forecast time; it was
   replaced by one that only uses an alert after it is closed, with a test.
6. **Text experiment.** Kept honest: a small, inconclusive gain is reported as such.

# Reproduce

Tested with Python 3.14 on Windows.

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
git clone --depth 1 https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset.git data_raw
python run.py                    # main result -> results.json
python -m pytest tests -q        # leakage tests
python fetch_tg.py 2026-08-01    # optional: collect Telegram posts -> data_raw/tg_kpszsu.csv
python run_text.py               # optional: text vs no-text comparison
```

The dataset repository is updated over time, so a fresh clone may contain data after 2026-10-11 and the test
period in the scripts is fixed to 2026-09-13 … 2026-10-11.

# Repository layout

```
src/
  data.py            load alerts, build hourly panel, long-alert flags
  features.py        lags, frequencies, cross-oblast features
  model.py           weekly walk-forward split and LightGBM
  text_features.py   Telegram text features
  grid.py            hourly grid helper used in exploration
tests/
  test_no_leakage.py leakage and split tests
exploration/         one-off checks: source comparison, drift, Lviv, baselines, diagnostics
run.py               main experiment
run_text.py          text experiment
fetch_tg.py          Telegram collection
```
