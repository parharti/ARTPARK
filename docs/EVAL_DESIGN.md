# EVAL_DESIGN.md

*Living document. Written as a design, updated as the data showed what was actually possible. Each section shows: Planned → Found → Adjusted.*

---

## 1. The decision this evaluation informs

- **Who uses the forecast:** a District Health Officer (DHO), every Monday.
- **What they do with it:** decide where to send fogging teams, hospital and platelet stock, and awareness campaigns for the coming weeks.
- **The forecast is one input, not the decision.**
- So the eval asks: *would the DHO be better off acting on this than ignoring it — and does he know when not to trust it?* It does not ask "what is the lowest error."

## 2. Why this evaluation can be trusted

- **Every metric is sliced by district.** The 6 districts range 10× in size — Bengaluru Urban alone would dominate any combined score. A single combined number hides the districts where the forecast actually fails.
- **The split respects time.** Forecasting is about predicting the future from the past, so the eval never splits the data randomly. It uses a rolling-origin simulator — walk forward one Monday at a time, the model only sees data from before that Monday.
- **The harness was checked before any modelling.** A from-scratch seasonal-naive matched the shipped baseline file exactly on all 588 well-formed rows. (The shipped file's other 36 rows were malformed — the horizon column didn't match the gap between issue and target week.) If the harness reproduces a known baseline exactly, it can be trusted to score new models fairly.

## 3. Data choices

- **Planned:** use the starter data; bring in public data if it's too thin.
- **Found:** 6 districts × 104 weeks (2023–24). Cases, weather, and a district registry. Clean — no missing weeks, ~29 missing rainfall values, and one reporting glitch in D03 around week 32 of 2024 (cases drop to 0 for a week, then spike).
- **Adjusted:** used the starter data as-is. No outside data. No tuning to its exact numbers — the brief says the submission is re-run on hidden data, so tuning would backfire. The D03 glitch was kept, not cleaned — it's real-world behaviour, and it became a probe.

## 4. How the data is split — rolling-origin

- **Planned:** train on 2023, test on 2024, moving forward one week at a time. At each Monday, forecast 2 and 4 weeks ahead for every district.
- **Found:** works. 48 issue-weeks × 6 districts × 2 horizons = 576 forecasts per model, each with the real outcome attached.
- **Why not a random split:** a random split lets the model train on weeks that come *after* the test weeks — it sees the future, so the score is fake.
- **Why not a single train/test split:** that trains the model once and freezes it — by late 2024 it's forecasting with knowledge a full year old. Rolling-origin lets the model see data right up to each forecast date, which is how it would actually be used. It also shows whether the forecast gets *worse* as the year goes on — a single split hides that.
- **Why stop at week 48, not 52:** a 4-week forecast issued after week 48 would target a week outside the data, with no real value to check it against. The longest horizon sets the cutoff.
- **Leakage guard — lag features:** "cases N weeks ago" is counted back from the forecast's *today* (the issue week), not from the week being predicted. That way every lag is a real, already-observed number, and means the same thing whether forecasting 2 or 4 weeks ahead. Counting back from the target week instead would pull in weeks that haven't happened yet.
- **Implementation note:** the simulator can retrain the model every week, but for this exercise the model was fit once on 2023 with fresh data fed in for the lag features at each step. A real deployment would retrain weekly — listed under limitations.

## 5. The modelling path — told honestly

- **Planned:** train a separate model per district, then combine them.
- **Found:** that gives only ~52 training weeks per district — too little to fit anything but a naive model without overfitting. Rejected before building.
- **Attempt 1 — one pooled regression (Poisson):** failed. The lag features meant different things during training and during prediction (the leakage issue in Section 4, before it was fixed). Worse than seasonal-naive.
- **Attempt 2 — seasonal-naive plus a learned correction:** failed for a structural reason. With only one prior year, the "seasonal average" for any week is just that one 2023 value — so the thing the correction was supposed to learn was always zero. This is a hard limit of a 2-year dataset, not a bug.
- **Final — pooled XGBoost:** cases converted to a rate per 100k (so the biggest district doesn't dominate); lag features fixed to count from the issue week; district added as a column; seasonal-naive's own prediction added as a feature. Marginally beat seasonal-naive overall, and clearly improved the districts where seasonal-naive failed worst — but didn't win everywhere.
- The failed attempts are reported, not hidden. They are evidence about what a 2-year dataset will and won't allow.

## 6. The metrics — and why each one

- **MAE** — the average size of the error, in cases. The headline "how wrong" number.
- **RMSE** — like MAE but punishes big misses harder. If RMSE is much larger than MAE, the model has occasional large blow-ups — and one missed surge week costs more than several small errors.
- **Bias** — the *direction* of the error. Under-predicting during a surge tells the DHO to relax when he should be mobilising. A model can have a low MAE and still be dangerous in one direction — only bias shows this.
- **Coverage** — does the 80% range actually contain the real value about 80% of the time? An accurate number with a dishonest range is still untrustworthy — the DHO needs to know how much to believe it.
- All four are reported **per-district, per-tier, per-horizon, and for the surge season alone (weeks 22–36)** — because the off-season is easy and makes every combined score look better than it is.

## 7. Per-district and per-tier — the main finding

- **Planned:** slice by tier (1, 2, 3), expecting tier-3 to be the weakest — that's what the brief's framing suggested.
- **Found:** the opposite. **Tier-3 was fine; tier-2 was the weakest.**
- **Why:** it's not about tier. It's about districts whose case numbers *changed between years*. Bengaluru Rural (D02, tier-2) had a 2024 surge about twice the size of its 2023 surge. Seasonal-naive assumes this year looks like last year, so it under-predicted D02 by about 52% every surge week. The failure follows the year-over-year change — which happened to land in tier-2.
- **Adjusted — what to actually do:** no single model is best everywhere, so the recommendation is a per-district assignment:
  - D01, D04, D06 → seasonal-naive (good enough, well-calibrated)
  - D02, D05 → XGBoost (clearly lower surge-season error)
  - D03 → neither is trustworthy; the reporting glitch corrupts it → keep officer review, add an anomaly flag
- This recommendation is only possible *because* the eval was sliced per-district. A combined-score eval would have shipped one model and quietly failed three districts.

## 8. What "good enough to pilot" means

A district is ready to pilot if **all three** are true:

1. The model beats seasonal-naive on that district's surge-season error — **or** matches it while being more robust when data goes bad.
2. It doesn't fail silently — its mistakes show up in the metrics and are explained.
3. Its uncertainty range is honest enough for the DHO to judge how much to trust it.

- **Verdict:** D01, D02, D04, D05, D06 are pilot-ready under the per-district assignment above. D03 is not ready as an automated forecast — officer review with an anomaly flag instead.
- The pilot is **staged**: the forecast goes in as an advisory signal with officer review kept, not as an automatic trigger.

## 9. Failure-mode probes

**Probe 1 — Reporting collapse (health-specific)**

- Real reporting drops during festivals, staff shortages, and PHC closures. The probe drops Hassan's (D04) reported cases by 40% for 3 weeks during peak surge, then re-runs both models.
- **Finding:** seasonal-naive was unaffected — it uses last year, not last week, so corrupted recent data can't reach it. XGBoost was thrown off for **6 weeks** — the corrupted weeks fed its lag features and dragged predictions low long after reporting recovered.
- **Response:** any data-quality alert should switch that district to seasonal-naive until the disruption is at least 6 weeks behind.

**Probe 2 — Outbreak-week performance**

- A forecast can look fine on average and still be useless on the weeks that matter. This probe scores the models only on weeks where cases jumped more than 50% from the week before.
- **Finding:** seasonal-naive held up (error of 18 on outbreak weeks — actually better than its 25 on quiet weeks). XGBoost was nearly 2× worse on outbreak weeks (43 vs 23) — its lag features can't see a sudden jump coming.
- **Together:** seasonal-naive is the more robust choice exactly when the DHO most needs a forecast — sudden surges and after data disruptions.

## 10. What was deliberately left out — and why

- **ARIMA / Prophet** — these need several years of seasonal history to learn the seasonal pattern. With one year in training, they'd likely just reproduce seasonal-naive without beating it.
- **Deep learning (LSTM and similar)** — 2 years × 6 districts is far too little data; it would overfit. The brief also doesn't score headline accuracy.
- **Weekly retraining** — the simulator supports it; for a 3-day exercise the model was fit once. Noted as a limitation, not hidden.
- **Tuning to the shipped data** — avoided on purpose, since the submission is re-run on hidden data.

## 11. Known limitations

- **2 years is not enough.** It allows one training year and one test year — nothing more. It can't support a learned seasonal correction, and it can't tell us whether 2024's patterns will hold in 2025. This is the biggest limit on the verdict.
- **The starter data is clean.** Real surveillance data has reporting lags, revisions, and missing weeks that are mostly absent here. Probe 1 simulates one such failure, but the model hasn't faced the full messiness of real data.
- **The uncertainty ranges are empirical** — built from training errors. Honest enough to be useful, but they'd improve with more history.
- **The verdict is limited by the data, not the model.** More years and realistic reporting-lag patterns would change what can be certified — and the report says so plainly.
