# REPORT.md

## Verdict

A weekly dengue forecast can be piloted in 5 of the 6 districts — but not with one single model, and not as an automatic trigger. We recommend a staged pilot: the forecast goes to the District Health Officer as an advisory signal, officer review stays in place, and each district uses the model that actually works for it. The sixth district, Mysuru (D03), is not yet ready, because a reporting problem in its data breaks the forecast — it should stay under officer judgment for now.

## What the forecast is for

Every Monday, a District Health Officer decides where to send fogging teams, hospital and platelet stock, and awareness campaigns for the coming weeks. The forecast predicts dengue cases 2 and 4 weeks ahead, per district, so resources can be moved before a surge instead of after. It is one input to the decision, not the decision itself.

## What we did

We took two years of weekly case data and split it to train and test a model. The important part was *how* we split it: by time, not at random. The model trains on 2023 and is tested on 2024 — so it can never accidentally see the future. That keeps the test honest.

## What we found

Dengue follows a strong seasonal pattern. Every district sees the same shape each year — quiet through spring, a sharp surge after the monsoon, a peak around August, a decline by November. Because this repeats so reliably, a very simple rule — "this year's week will look like last year's same week" — is already accurate. It is genuinely hard to beat.

But every district has its own pattern. The simple seasonal rule works well for Bengaluru Urban, Hassan, and Mandya. It fails where a district's case numbers shifted between years — Bengaluru Rural's 2024 surge was about twice its 2023 surge, and the seasonal rule under-predicted it by roughly half, every surge week.

The natural fix is to train a separate model for each district. But each district only has about a year of training data — too little to learn from without overfitting. So we tried a shared model across all districts instead. That failed, and a second attempt — learning a correction on top of the seasonal rule — failed too. Both failed for the same underlying reason: with only one prior year, there is simply not enough history to learn from. This is a limit of the data, not the method.

A machine-learning model (XGBoost) did help — it fixed the districts the seasonal rule failed worst (Bengaluru Rural, Tumakuru) — but it was less reliable on the others. No single model was best everywhere.

One more finding worth noting: we expected the smaller, rural districts to be hardest. They were not. The weakest group was the tier-2 districts, and the real driver of failure was year-over-year change in case numbers — not district size.

## Recommendation — per district

| District | Use this model | Status |
|---|---|---|
| Bengaluru Urban (D01) | Seasonal rule | Pilot-ready |
| Bengaluru Rural (D02) | XGBoost | Pilot-ready |
| Mysuru (D03) | Neither — officer review + data-quality flag | Not ready |
| Hassan (D04) | Seasonal rule | Pilot-ready |
| Tumakuru (D05) | XGBoost | Pilot-ready |
| Mandya (D06) | Seasonal rule | Pilot-ready |

## What to watch out for

We ran two stress tests to find when the forecast should not be trusted.

When reporting breaks down — during festivals, staff shortages, or PHC closures — the machine-learning model stays wrong for about 6 weeks afterwards, because it learns from recent case numbers. The seasonal rule is not affected by this. So any data-quality alert should switch that district back to the seasonal rule until reporting has been stable for 6 weeks.

During sudden surges — weeks where cases jump more than 50% — the seasonal rule holds up better than the machine-learning model. That is exactly the moment the officer most needs a reliable signal, and it is a key reason the pilot keeps officer review in place.

## What would change this verdict

More data. Three or more years would allow a model that genuinely improves on the seasonal rule. Realistic reporting-lag data would let us test the forecast against the messiness of real surveillance systems. Until then, a staged pilot — advisory forecast, officer review retained, the right model chosen per district — is the responsible recommendation.
