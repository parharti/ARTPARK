# DATA_EXPLORATION.md

*First look at the starter data, before any modelling. The purpose of this step is to understand what the data actually contains, judge what is and isn't learnable from it, and surface any quality issues that the evaluation must account for.*

---

## 1. What the dataset is

- **6 districts**, modelled on Karnataka — Bengaluru Urban, Bengaluru Rural, Mysuru, Hassan, Tumakuru, Mandya.
- **104 weeks** of weekly data per district — 2023-W01 to 2024-W52 (two full years).
- **Three input tables:** weekly dengue case counts, weekly weather (rainfall, temperature, humidity), and a district registry (population, urbanisation score, tier).
- A shipped baseline submission (`baseline_seasonal_naive.csv`) showing the required output format.

## 2. Data quality check

![Data summary](figures/data_summary.png)

- **104 weeks per district, no missing weeks.** The panel is complete on the time axis — no gaps to interpolate.
- **Cases, temperature, humidity, district metadata: zero missing values.**
- **Rainfall: 29 missing values** out of 624 rows (~4.6%). Manageable — handled by median fill within district at feature-build time.
- **Date range parsed cleanly:** 2023-01-02 to 2024-12-23, ISO weeks aligned to Mondays.

**Conclusion:** the data is clean and internally consistent. The only quality issue in the raw tables is missing rainfall, which is small and handled. The more important issue is a behavioural anomaly visible only in the plot — see Finding 4.

## 3. Weekly cases by district

![Weekly dengue cases by district](figures/cases_by_district.png)

*Each panel is one district. The red dashed line marks the 2023 → 2024 boundary — left of it is the training year, right of it is the test year.*

### Finding 1 — Strong, consistent seasonality

Every district shows the same shape in both years: a quiet baseline through ~week 20, a sharp climb from ~week 22, a peak around weeks 30–34, and a decline back to baseline by November. The seasonal signal is the dominant feature of this data.

**Implication for the eval:** seasonal-naive ("predict this week = same week last year") will be a strong baseline, because the seasonal pattern genuinely repeats. Beating it requires capturing what it *misses* — year-over-year amplitude shifts, not the seasonal shape itself.

### Finding 2 — A 10× scale gap between districts

- D01 (Bengaluru Urban): cases range ~500 to ~2000.
- D02 (Bengaluru Rural): ~20 to ~230.
- D06 (Mandya): ~20 to ~140.

**Implication for the eval:** a single model trained on raw counts would let D01 dominate the loss and ignore the small districts. This forces a decision — model the rate per 100k population, not raw counts — and it forces per-district metric reporting, since an aggregate score is just D01's score in disguise.

### Finding 3 — 2024 peaks differ from 2023, district by district

- D02 (Bengaluru Rural): 2024 peak ~230 vs 2023 peak ~125 — **roughly 2× larger.**
- D05 (Tumakuru): 2024 peak ~200 vs 2023 peak ~270 — **smaller this year.**
- D01, D03, D04, D06: peaks broadly similar between years.

**Implication for the eval:** this is exactly where seasonal-naive fails — it assumes 2024 resembles 2023. In D02 it will badly under-predict; in D05 it will over-predict. The failure tracks *which districts changed between years*, not district size or tier. This finding shaped the per-district subgroup analysis.

### Finding 4 — A reporting anomaly in D03 (Mysuru), 2024

In the D03 panel, around week 32 of 2024, cases collapse from ~280 to 0 in a single week, then spike back to ~370 the next week. This is not real dengue behaviour — it is almost certainly a reporting failure (a holiday, staff disruption, or PHC submission delay).

**Implication for the eval:**
- The anomaly is kept in the data, not smoothed — it is real-world behaviour and the model must be evaluated against it honestly.
- It became the basis of a failure-mode probe (the reporting-collapse probe), which deliberately recreates this kind of disruption to test model robustness.
- It explains, in advance, why any model relying on recent-case features is likely to struggle on D03 in the test year.

## 4. What this exploration locked in

Before writing a line of model code, the exploration fixed five evaluation decisions:

1. **Model the rate per 100k, not raw counts** — so the largest district does not dominate.
2. **Report every metric per-district and per-tier** — aggregate scores hide failure.
3. **Use seasonal-naive as the baseline to beat** — the seasonality is real and strong.
4. **Expect the hard cases to be year-over-year amplitude shifts** — not the seasonal shape.
5. **Keep the D03 anomaly in the data and probe it** — rather than cleaning it away.

These decisions are carried forward into `EVAL_DESIGN.md`.
