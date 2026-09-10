# Downstream evaluation · loss24-muon-only-k4-s42-20260720 · muon

Task-native metrics are reported separately; no cross-task mean is computed.

## Status

| Status | Count |
|---|---:|
| evaluated | 24 |

## Results

| Setting | Task | Method | Primary metric | Value |
|---|---|---|---|---:|
| alpaca_samsum | samsum | FullTraining | rougeL | 0.089415 |
| alpaca_samsum | samsum | GlobalRaw | rougeL | 0.085147 |
| alpaca_samsum | samsum | GlobalSoft | rougeL | 0.087529 |
| alpaca_samsum | samsum | LayerwiseMuonMatrixSur | rougeL | 0.088599 |
| alpaca_samsum | samsum | LayerwiseRaw | rougeL | 0.085658 |
| alpaca_samsum | samsum | LayerwiseSoft | rougeL | 0.090290 |
| less_squad | squad | FullTraining | f1_score | 9.599693 |
| less_squad | squad | GlobalRaw | f1_score | 9.052688 |
| less_squad | squad | GlobalSoft | f1_score | 9.607747 |
| less_squad | squad | LayerwiseMuonMatrixSur | f1_score | 9.071209 |
| less_squad | squad | LayerwiseRaw | f1_score | 9.496770 |
| less_squad | squad | LayerwiseSoft | f1_score | 9.431274 |
| less_tydiqa | tydiqa | FullTraining | f1_score | 11.457334 |
| less_tydiqa | tydiqa | GlobalRaw | f1_score | 11.746766 |
| less_tydiqa | tydiqa | GlobalSoft | f1_score | 12.088823 |
| less_tydiqa | tydiqa | LayerwiseMuonMatrixSur | f1_score | 11.009224 |
| less_tydiqa | tydiqa | LayerwiseRaw | f1_score | 11.544008 |
| less_tydiqa | tydiqa | LayerwiseSoft | f1_score | 11.384461 |
| triviaqa_nq | nq_open | FullTraining | f1_score | 6.878433 |
| triviaqa_nq | nq_open | GlobalRaw | f1_score | 7.442942 |
| triviaqa_nq | nq_open | GlobalSoft | f1_score | 8.080775 |
| triviaqa_nq | nq_open | LayerwiseMuonMatrixSur | f1_score | 7.024179 |
| triviaqa_nq | nq_open | LayerwiseRaw | f1_score | 7.211776 |
| triviaqa_nq | nq_open | LayerwiseSoft | f1_score | 7.163196 |
