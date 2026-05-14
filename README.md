# Dengue district-level forecasting (Pathway B)

Weekly dengue case forecasts for 6 districts of Karnataka, 2 and 4 weeks ahead.
Built for a District Health Officer (DHO) making weekly resourcing decisions —
fogging teams, platelet stock, awareness campaigns. The forecast is an input,
not the decision.

## Layout

```
.
├── README.md
├── requirements.txt
├── run.py                      Single-command pipeline: harness check + every model + metrics + CSVs
├── data/                       Raw weekly panels (cases, weather, district registry)
├── docs/
│   ├── REPORT.md               Plain-English verdict and per-district recommendation — start here
│   ├── PATHWAY_RATIONALE.md    Why Pathway B
│   ├── DATA_EXPLORATION.md     First look at the data; decisions locked in before modelling
│   ├── EVAL_DESIGN.md          Full evaluation design — metrics, slices, probes, limits
│   ├── AI_USAGE.md             Honest account of where AI was used, accepted, rejected, verified
│   └── flowchart.png           Pipeline diagram
├── notebooks/
│   └── dengue_forecasting.ipynb   Thin driver: imports src/, runs the harness, prints tables
├── src/forecasting/
│   ├── data.py                 Loaders and panel construction
│   ├── features.py             Feature builders (Poisson lags, XGBoost lags)
│   ├── forecasters.py          Forecaster ABC + all 7 model implementations
│   ├── simulator.py            Rolling-origin simulator (run_simulation)
│   ├── metrics.py              MAE / RMSE / bias / 80%-coverage, headline and grouped
│   └── probes.py               Failure-mode probes (reporting collapse, outbreak weeks)
└── submissions/
    └── baseline_seasonal_naive.csv   The shipped seasonal-naive baseline
```

## Run it

One command runs the whole pipeline — harness verification, every model, metrics, and CSVs:

```bash
pip install -r requirements.txt
python run.py
```

Outputs are written to `submissions/<model_name>.csv`. To run a subset of models,
or to skip the failure-mode probes:

```bash
python run.py --models seasonal_naive xgboost_with_sn
python run.py --no-probes
```

Or, for the same flow as a notebook:

```bash
jupyter notebook notebooks/dengue_forecasting.ipynb
```

The model and harness code is in `src/forecasting/` and is intended to be re-used
and swapped. The simulator only ever calls `.fit()` and `.predict()` on a
`Forecaster`, so adding a new model is a single-class change.

## Where to read what

Suggested reading order:

1. **`docs/REPORT.md`** — plain-English verdict, per-district pilot recommendation,
   and what would change the verdict. Start here.
2. **`docs/PATHWAY_RATIONALE.md`** — why I chose Pathway B and the one concrete risk
   the design guards against.
3. **`docs/DATA_EXPLORATION.md`** — first look at the data and the five evaluation
   decisions that were locked in before any modelling.
4. **`docs/EVAL_DESIGN.md`** — the full evaluation design: rolling-origin split,
   metrics and why each one, per-district findings, failure-mode probes, what was
   deliberately left out, and known limitations.
5. **`docs/AI_USAGE.md`** — honest account of where AI was used, what I accepted,
   what I rejected, and how the work was verified.
6. **`docs/flowchart.png`** — pipeline diagram.
7. **`notebooks/dengue_forecasting.ipynb`** — minimal end-to-end run that reproduces
   the numbers cited in `EVAL_DESIGN.md`.

## Models

Tried in order, all in `src/forecasting/forecasters.py`:

| name | notes |
|---|---|
| `LastWeekNaive` | persistence baseline |
| `SeasonalNaive` | cases at (target − 52w); reproduces the shipped baseline exactly on its 588 well-formed rows |
| `PoissonForecaster` | attempt 1 — beaten by seasonal-naive |
| `SeasonalNaivePlus` / `V2` | attempt 2 — failed for a structural reason (see EVAL_DESIGN §5) |
| `XGBoostForecaster` | pooled, rate-per-100k, one-hot district |
| `XGBoostForecasterV2` | adds seasonal-naive prediction as a feature; marginally beats SN overall, clearly wins on D02 and D05 |

The recommendation is **per-district model assignment**, not one winner — see
EVAL_DESIGN §7.

## Failure-mode probes

Two probes are part of the deliverable and run by default at the end of `python run.py`.
Full description in `docs/EVAL_DESIGN.md` §9.

**Probe 1 — reporting collapse.** Drops Hassan's (D04) reported cases by 40% for its
3 highest surge weeks, then re-runs the candidate models. Predictions are scored against
the **true** actuals from the clean panel (not the corrupted ones), so the probe measures
how the corrupted inputs degrade the model — not the trivial fact that the actuals moved.

- `SeasonalNaive` is unaffected (delta MAE = +0.00) — it predicts from last year's same
  week, so corrupted recent data can't reach it.
- `XGBoostForecasterV2` shows persistent error and does not recover within a 6-week
  window after the last corrupt week — its lag features carry the corruption forward.
- **Operational implication:** any data-quality alert should switch that district to
  seasonal-naive until the disruption is at least 6 weeks behind.

**Probe 2 — outbreak weeks.** Scores every model only on target weeks where actuals
jumped > 50% from the prior week (the weeks that actually matter for DHO decisions).

- `SeasonalNaive` outbreak / quiet MAE ratio ≈ 0.75 (actually *better* on outbreaks).
- `XGBoostForecasterV2` outbreak / quiet MAE ratio ≈ 2× (nearly twice as bad on
  outbreak weeks).
- **Operational implication:** seasonal-naive is the more robust choice exactly when
  the DHO most needs a forecast — sudden surges and after data disruptions.
