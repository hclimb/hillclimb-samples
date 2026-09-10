# Downstream evaluation · baseline9-layerwise-s43-20260721 · muon

Task-native metrics are reported separately; no cross-task mean is computed.

## Status

| Status | Count |
|---|---:|
| evaluated | 32 |

## Results

| Setting | Task | Method | Primary metric | Value |
|---|---|---|---|---:|
| alpaca_samsum | samsum | FullTraining | rougeL | 0.079808 |
| alpaca_samsum | samsum | LayerwiseMuonMatrixSur | rougeL | 0.080147 |
| alpaca_samsum | samsum | LayerwiseMuonOnlyPSur | rougeL | 0.080420 |
| alpaca_samsum | samsum | LayerwiseMuonOnlySatPSur | rougeL | 0.080028 |
| alpaca_samsum | samsum | LayerwiseMuonOnlySatSur | rougeL | 0.080756 |
| alpaca_samsum | samsum | LayerwiseRaw | rougeL | 0.079447 |
| alpaca_samsum | samsum | LayerwiseSoft | rougeL | 0.080499 |
| alpaca_samsum | samsum | LayerwiseSoftP | rougeL | 0.076103 |
| less_squad | squad | FullTraining | f1_score | 9.343967 |
| less_squad | squad | LayerwiseMuonMatrixSur | f1_score | 9.501065 |
| less_squad | squad | LayerwiseMuonOnlyPSur | f1_score | 9.417981 |
| less_squad | squad | LayerwiseMuonOnlySatPSur | f1_score | 9.220817 |
| less_squad | squad | LayerwiseMuonOnlySatSur | f1_score | 9.398778 |
| less_squad | squad | LayerwiseRaw | f1_score | 9.211806 |
| less_squad | squad | LayerwiseSoft | f1_score | 9.448162 |
| less_squad | squad | LayerwiseSoftP | f1_score | 9.479881 |
| less_tydiqa | tydiqa | FullTraining | f1_score | 11.694752 |
| less_tydiqa | tydiqa | LayerwiseMuonMatrixSur | f1_score | 12.460003 |
| less_tydiqa | tydiqa | LayerwiseMuonOnlyPSur | f1_score | 11.121560 |
| less_tydiqa | tydiqa | LayerwiseMuonOnlySatPSur | f1_score | 11.355313 |
| less_tydiqa | tydiqa | LayerwiseMuonOnlySatSur | f1_score | 11.171075 |
| less_tydiqa | tydiqa | LayerwiseRaw | f1_score | 12.894648 |
| less_tydiqa | tydiqa | LayerwiseSoft | f1_score | 11.904339 |
| less_tydiqa | tydiqa | LayerwiseSoftP | f1_score | 11.768463 |
| triviaqa_nq | nq_open | FullTraining | f1_score | 7.408699 |
| triviaqa_nq | nq_open | LayerwiseMuonMatrixSur | f1_score | 7.390590 |
| triviaqa_nq | nq_open | LayerwiseMuonOnlyPSur | f1_score | 7.173031 |
| triviaqa_nq | nq_open | LayerwiseMuonOnlySatPSur | f1_score | 7.330879 |
| triviaqa_nq | nq_open | LayerwiseMuonOnlySatSur | f1_score | 7.130136 |
| triviaqa_nq | nq_open | LayerwiseRaw | f1_score | 7.126431 |
| triviaqa_nq | nq_open | LayerwiseSoft | f1_score | 6.694960 |
| triviaqa_nq | nq_open | LayerwiseSoftP | f1_score | 8.121061 |
