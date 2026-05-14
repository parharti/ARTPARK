# AI_USAGE.md

AI-assisted coding and discussion was used throughout this submission. The brief
permits and expects this. What follows is an honest account of where AI was used,
what I accepted, what I rejected, and how I verified the work.

## Context

This was done in roughly 72 hours alongside a full-time internship. Given that, I
used an AI assistant the way I would use a teammate I could talk to at any hour —
to think through the pipeline out loud, to get coding assistance, and to handle
the mechanical work of restructuring code and drafting documents. The important
point is that the conversation was always back-and-forth. It was never one-sided:
I directed it, questioned it, and made the decisions. AI sped up the execution; it
did not replace the thinking.

## Tools used

- **Conversational AI assistant** — a thinking partner for ideation, and for
  writing notebook code in an iterative, cell-by-cell exchange.
- **Google Colab** — where all the code was actually run and checked.
- **Claude Code** — used to restructure the verified notebook into a reproducible
  repo (src/ files, config, single-command run) and to draft README.md.

## Where AI was used

**Ideation and design.** I discussed the problem with the AI to work out how to
build a full pipeline — choosing Path B, understanding what time-series
forecasting requires that ordinary prediction does not, deciding to evaluate
per-district rather than only in aggregate, choosing the metrics, and defining
what "good enough to pilot" should mean. The framing and the decisions are mine;
the AI helped me pressure-test them.

**Coding.** With AI now able to handle a lot of the mechanical coding, I used it
for that — but cell by cell, running each one in Colab, reading the output, and
deciding it was correct before moving on. It was an incremental build, not a
single generated script.

**Documents and repo structure.** AI was used to draft the documents and to lay
out the repo cleanly. This was done with full awareness of the content — the
ideas, the decisions, and the findings being written up are ones I worked through
myself.

## What I accepted

- The rolling-origin simulator design — walk forward week by week, model sees only
  past data.
- The Forecaster interface — every model exposes the same fit/predict, so swapping
  the model is a config change, not a code change.
- The metric set, and slicing every metric by district, tier, horizon, and season.
- The two failure-mode probes (reporting collapse, outbreak-week performance).

## What I rejected or changed

- I pushed back on training a separate model per district — correctly, as the data
  turned out to be too thin for it.
- I questioned an early date-format choice and a feature-leakage issue in the lag
  features; both were corrected.
- When the AI estimated a verification count from memory, I insisted on computing
  it exactly — which surfaced that 588 of 624 baseline rows were well-formed, not
  the rough figure first given.
- I rejected chasing a model that beat seasonal-naive once it was clear the 2-year
  dataset structurally limited what was learnable, and redirected effort to the
  evaluation, where the brief's weight lies.

## How I verified

- The seasonal-naive implementation was checked against the shipped baseline file
  before any modelling — it matched on all 588 well-formed rows.
- Every model was run through the same simulator and scored with the same metrics,
  so comparisons are like-for-like.
- Outputs were inspected at each step in Colab — forecast counts, missing values,
  per-district numbers — not trusted blindly.
- The two failed model attempts (pooled Poisson, learned residual correction) are
  reported, not hidden, because understanding why they failed was part of
  verifying what the data can and cannot support.

## Interview note

I can explain and modify the code without AI assistance. The pipeline was built
and verified incrementally, and the design decisions in EVAL_DESIGN.md are ones I
made and can defend.
