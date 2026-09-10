# Downstream evaluation · baseline9-layerwise-s43-20260721 · adamw

Task-native metrics are reported separately; no cross-task mean is computed.

## Status

| Status | Count |
|---|---:|
| evaluated | 20 |

## Results

| Setting | Task | Method | Primary metric | Value |
|---|---|---|---|---:|
| alpaca_samsum | samsum | FullTraining | rougeL | 0.162263 |
| alpaca_samsum | samsum | LayerwiseOptA | rougeL | 0.166629 |
| alpaca_samsum | samsum | LayerwiseRaw | rougeL | 0.162160 |
| alpaca_samsum | samsum | LayerwiseSoft | rougeL | 0.172484 |
| alpaca_samsum | samsum | LayerwiseSoftP | rougeL | 0.172901 |
| less_squad | squad | FullTraining | f1_score | 9.154012 |
| less_squad | squad | LayerwiseOptA | f1_score | 8.911619 |
| less_squad | squad | LayerwiseRaw | f1_score | 7.849267 |
| less_squad | squad | LayerwiseSoft | f1_score | 8.283855 |
| less_squad | squad | LayerwiseSoftP | f1_score | 8.718503 |
| less_tydiqa | tydiqa | FullTraining | f1_score | 6.611433 |
| less_tydiqa | tydiqa | LayerwiseOptA | f1_score | 20.153611 |
| less_tydiqa | tydiqa | LayerwiseRaw | f1_score | 21.592614 |
| less_tydiqa | tydiqa | LayerwiseSoft | f1_score | 20.608395 |
| less_tydiqa | tydiqa | LayerwiseSoftP | f1_score | 32.782806 |
| triviaqa_nq | nq_open | FullTraining | f1_score | 12.594632 |
| triviaqa_nq | nq_open | LayerwiseOptA | f1_score | 14.305108 |
| triviaqa_nq | nq_open | LayerwiseRaw | f1_score | 15.138442 |
| triviaqa_nq | nq_open | LayerwiseSoft | f1_score | 14.393680 |
| triviaqa_nq | nq_open | LayerwiseSoftP | f1_score | 16.154341 |
