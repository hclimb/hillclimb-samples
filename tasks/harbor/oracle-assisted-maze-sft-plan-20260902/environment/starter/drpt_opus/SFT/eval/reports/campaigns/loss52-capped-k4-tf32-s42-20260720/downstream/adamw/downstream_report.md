# Downstream evaluation · loss52-capped-k4-tf32-s42-20260720 · adamw

Task-native metrics are reported separately; no cross-task mean is computed.

## Status

| Status | Count |
|---|---:|
| evaluated | 28 |

## Results

| Setting | Task | Method | Primary metric | Value |
|---|---|---|---|---:|
| alpaca_samsum | samsum | FullTraining | rougeL | 0.154038 |
| alpaca_samsum | samsum | GlobalOptA | rougeL | 0.172007 |
| alpaca_samsum | samsum | GlobalRaw | rougeL | 0.159929 |
| alpaca_samsum | samsum | GlobalSoft | rougeL | 0.178218 |
| alpaca_samsum | samsum | LayerwiseOptA | rougeL | 0.191645 |
| alpaca_samsum | samsum | LayerwiseRaw | rougeL | 0.190263 |
| alpaca_samsum | samsum | LayerwiseSoft | rougeL | 0.192001 |
| less_squad | squad | FullTraining | f1_score | 8.967799 |
| less_squad | squad | GlobalOptA | f1_score | 8.815712 |
| less_squad | squad | GlobalRaw | f1_score | 8.718764 |
| less_squad | squad | GlobalSoft | f1_score | 8.670452 |
| less_squad | squad | LayerwiseOptA | f1_score | 8.847162 |
| less_squad | squad | LayerwiseRaw | f1_score | 8.639045 |
| less_squad | squad | LayerwiseSoft | f1_score | 9.099751 |
| less_tydiqa | tydiqa | FullTraining | f1_score | 6.684564 |
| less_tydiqa | tydiqa | GlobalOptA | f1_score | 8.993642 |
| less_tydiqa | tydiqa | GlobalRaw | f1_score | 7.624396 |
| less_tydiqa | tydiqa | GlobalSoft | f1_score | 8.524920 |
| less_tydiqa | tydiqa | LayerwiseOptA | f1_score | 18.477986 |
| less_tydiqa | tydiqa | LayerwiseRaw | f1_score | 17.504522 |
| less_tydiqa | tydiqa | LayerwiseSoft | f1_score | 16.621123 |
| triviaqa_nq | nq_open | FullTraining | f1_score | 12.503642 |
| triviaqa_nq | nq_open | GlobalOptA | f1_score | 12.370639 |
| triviaqa_nq | nq_open | GlobalRaw | f1_score | 13.092068 |
| triviaqa_nq | nq_open | GlobalSoft | f1_score | 13.391775 |
| triviaqa_nq | nq_open | LayerwiseOptA | f1_score | 14.578918 |
| triviaqa_nq | nq_open | LayerwiseRaw | f1_score | 14.683203 |
| triviaqa_nq | nq_open | LayerwiseSoft | f1_score | 14.129634 |
