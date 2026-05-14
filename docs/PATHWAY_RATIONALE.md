Chosen pathway: B — District-level dengue forecasting.

Why Path B
I could have done Path A comfortably. I have recent experience with exactly what it needs RAG, LLMs, Sarvam for multilingual Indic support, and RASA for intent-control to keep an LLM grounded. I have shipped a multilingual health chatbot before.
I chose Path B because the problem shape was new to me. I have built prediction systems ,
including weather-driven Random Forest models at ISRO but not time-series forecasting specifically.
Forecasting demands a different discipline: rolling-origin evaluation, never splitting data randomly, and deciding what "good enough to pilot" means for a District Health Officer who acts on the forecast, not a researcher who reads it.

What I give up by not picking Path A
Path A is a proven skill for me, picking it would mean re-proving what I can already do. By choosing Path B I give that up on purpose.

One concrete risk, and how the design guards against it
The risk of picking a new topic is that prediction habits leak into a forecasting problem. I have built prediction systems but not forecasts, so my instinct could be to split data randomly, or to build features without checking whether they use information that wouldn't be available at forecast time. In time-series, that is data leakage: the model sees the future during training, the evaluation looks excellent, and the pilot fails in the field.
The design guards against this explicitly time-ordered splits only, a rolling-origin simulator that walks week by week and shows the model only past data, and every feature computed strictly from history available at forecast time. The seasonal-naive baseline was verified against the shipped file (588 well-formed rows) before any modelling, so the evaluation harness itself is trustworthy.