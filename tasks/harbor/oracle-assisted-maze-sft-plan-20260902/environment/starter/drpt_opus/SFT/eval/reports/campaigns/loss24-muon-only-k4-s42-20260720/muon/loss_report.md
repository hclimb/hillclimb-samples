# AdamW/Muon loss-curve comparison

Successful requested runs: **24 / 24**. Missing or incomplete: **0**.

## Reading the metrics

- **Target validation** is `val_loss` on the small target/proxy set used by the selection method. Lower final loss and lower normalized AUC are better, but this set is not independent of selection.
- **General held-out** is `eval_loss` on the larger evaluation set. It is the main loss-based generalization cross-check.
- **Train** compares the mean of the first and last 10% of logged steps; plots use a centered rolling mean. For merged-batch selection runs this is the returned train+target merged loss, while FullTraining returns train-only loss, so cross-method absolute train-loss ranks are not apples-to-apples.
- AUC integrates loss over normalized progress from 0 to 1, making it comparable across run lengths. Lower is better. Reduction is initial minus final, so positive is improvement.
- Both `muon` and legacy `hybrid` runs use the `HybridMuonAdamW` runtime: Muon updates eligible matrix parameters and AdamW updates embeddings, norms, biases, and other ineligible parameters. The `muon` label uses only Muon-managed matrices for the spectral surrogate score; `hybrid` preserves mixed Muon+AdamW scoring.

## Winners by setting

| Setting | Optimizer | Lowest target final | Lowest target AUC | Lowest general final | Lowest general AUC | Largest within-run train reduction |
|---|---|---|---|---|---|---|
| alpaca_samsum | Muon | GlobalSoft | GlobalSoft | GlobalSoft | GlobalSoft | LayerwiseMuonMatrixSur |
| less_squad | Muon | LayerwiseRaw | LayerwiseRaw | LayerwiseRaw | LayerwiseRaw | FullTraining |
| less_tydiqa | Muon | LayerwiseRaw | LayerwiseRaw | LayerwiseRaw | LayerwiseRaw | FullTraining |
| triviaqa_nq | Muon | FullTraining | FullTraining | FullTraining | FullTraining | FullTraining |

## Win counts

Counts are across the four settings; ties count for every tied method. Raw losses are never averaged across different tasks.

| Optimizer | Method | Target final | Target AUC | General final | General AUC | Available settings |
|---|---|---:|---:|---:|---:|---:|
| Muon | FullTraining | 1 | 1 | 1 | 1 | 4 |
| Muon | GlobalRaw | 0 | 0 | 0 | 0 | 4 |
| Muon | LayerwiseRaw | 2 | 2 | 2 | 2 | 4 |
| Muon | LayerwiseMuonMatrixSur | 0 | 0 | 0 | 0 | 4 |
| Muon | GlobalSoft | 1 | 1 | 1 | 1 | 4 |
| Muon | LayerwiseSoft | 0 | 0 | 0 | 0 | 4 |

## Detailed loss reductions

### alpaca_samsum · Muon

| Method | Target initial → final (reduction; AUC) | General initial → final (reduction; AUC) | Train early → late (reduction; AUC) | W&B |
|---|---:|---:|---:|---|
| FullTraining | 2.2138 → 2.0090 (0.2047; 2.0352) | 2.0607 → 1.9483 (0.1124; 1.9612) | 1.7044 → 1.6354 (0.0690; 1.6472) | [run](https://wandb.ai/leena12/drpt_opus/runs/hi449cdx) |
| GlobalRaw | 2.2138 → 2.0173 (0.1965; 2.0417) | 2.0607 → 1.9497 (0.1109; 1.9631) | 1.7233 → 1.6598 (0.0634; 1.6708) | [run](https://wandb.ai/leena12/drpt_opus/runs/042vj9hr) |
| LayerwiseRaw | 2.2138 → 1.9929 (0.2208; 2.0224) | 2.0607 → 1.9367 (0.1239; 1.9523) | 1.7244 → 1.6611 (0.0634; 1.6718) | [run](https://wandb.ai/leena12/drpt_opus/runs/dxzwzbwc) |
| LayerwiseMuonMatrixSur | 2.2138 → 2.0086 (0.2051; 2.0338) | 2.0607 → 1.9463 (0.1143; 1.9602) | 1.7201 → 1.6509 (0.0693; 1.6625) | [run](https://wandb.ai/leena12/drpt_opus/runs/726orc5z) |
| GlobalSoft | 2.2138 → 1.9847 (0.2291; 2.0132) | 2.0607 → 1.9320 (0.1286; 1.9478) | 1.7223 → 1.6568 (0.0655; 1.6675) | [run](https://wandb.ai/leena12/drpt_opus/runs/5zl2g6r1) |
| LayerwiseSoft | 2.2138 → 1.9902 (0.2236; 2.0190) | 2.0607 → 1.9364 (0.1242; 1.9514) | 1.7234 → 1.6594 (0.0640; 1.6703) | [run](https://wandb.ai/leena12/drpt_opus/runs/0y7urae3) |

### less_squad · Muon

| Method | Target initial → final (reduction; AUC) | General initial → final (reduction; AUC) | Train early → late (reduction; AUC) | W&B |
|---|---:|---:|---:|---|
| FullTraining | 4.1166 → 3.8233 (0.2933; 3.8767) | 4.8667 → 4.5712 (0.2954; 4.6263) | 1.9665 → 1.8579 (0.1086; 1.8839) | [run](https://wandb.ai/leena12/drpt_opus/runs/wb2zv4vy) |
| GlobalRaw | 4.1166 → 3.8063 (0.3103; 3.8626) | 4.8667 → 4.5565 (0.3102; 4.6174) | 1.9973 → 1.8981 (0.0993; 1.9221) | [run](https://wandb.ai/leena12/drpt_opus/runs/crwhoe4t) |
| LayerwiseRaw | 4.1166 → 3.7868 (0.3298; 3.8495) | 4.8667 → 4.5349 (0.3318; 4.6011) | 1.9980 → 1.8991 (0.0990; 1.9231) | [run](https://wandb.ai/leena12/drpt_opus/runs/xovfjabo) |
| LayerwiseMuonMatrixSur | 4.1166 → 3.8055 (0.3111; 3.8669) | 4.8667 → 4.5563 (0.3104; 4.6146) | 1.9953 → 1.8878 (0.1075; 1.9135) | [run](https://wandb.ai/leena12/drpt_opus/runs/fn9brzxb) |
| GlobalSoft | 4.1166 → 3.7910 (0.3256; 3.8552) | 4.8667 → 4.5441 (0.3226; 4.6065) | 1.9979 → 1.8974 (0.1005; 1.9216) | [run](https://wandb.ai/leena12/drpt_opus/runs/2f6e2b3v) |
| LayerwiseSoft | 4.1166 → 3.7884 (0.3282; 3.8502) | 4.8667 → 4.5416 (0.3251; 4.6017) | 1.9979 → 1.8981 (0.0998; 1.9225) | [run](https://wandb.ai/leena12/drpt_opus/runs/nu9gz6kg) |

### less_tydiqa · Muon

| Method | Target initial → final (reduction; AUC) | General initial → final (reduction; AUC) | Train early → late (reduction; AUC) | W&B |
|---|---:|---:|---:|---|
| FullTraining | 1.4262 → 1.3211 (0.1050; 1.3417) | 1.9643 → 1.8158 (0.1485; 1.8433) | 1.9664 → 1.8580 (0.1084; 1.8839) | [run](https://wandb.ai/leena12/drpt_opus/runs/1kowxfot) |
| GlobalRaw | 1.4262 → 1.3185 (0.1077; 1.3387) | 1.9643 → 1.8080 (0.1563; 1.8387) | 1.9401 → 1.8401 (0.0999; 1.8632) | [run](https://wandb.ai/leena12/drpt_opus/runs/aan8j63j) |
| LayerwiseRaw | 1.4262 → 1.3036 (0.1226; 1.3303) | 1.9643 → 1.7917 (0.1726; 1.8272) | 1.9404 → 1.8395 (0.1009; 1.8631) | [run](https://wandb.ai/leena12/drpt_opus/runs/pzrqz2sv) |
| LayerwiseMuonMatrixSur | 1.4262 → 1.3192 (0.1069; 1.3372) | 1.9643 → 1.8062 (0.1581; 1.8364) | 1.9377 → 1.8298 (0.1080; 1.8545) | [run](https://wandb.ai/leena12/drpt_opus/runs/2ysnamiq) |
| GlobalSoft | 1.4262 → 1.3070 (0.1192; 1.3329) | 1.9643 → 1.7959 (0.1685; 1.8301) | 1.9398 → 1.8381 (0.1017; 1.8618) | [run](https://wandb.ai/leena12/drpt_opus/runs/nr46w0hl) |
| LayerwiseSoft | 1.4262 → 1.3122 (0.1139; 1.3324) | 1.9643 → 1.7983 (0.1660; 1.8296) | 1.9404 → 1.8390 (0.1015; 1.8626) | [run](https://wandb.ai/leena12/drpt_opus/runs/i9rslg6d) |

### triviaqa_nq · Muon

| Method | Target initial → final (reduction; AUC) | General initial → final (reduction; AUC) | Train early → late (reduction; AUC) | W&B |
|---|---:|---:|---:|---|
| FullTraining | 4.6689 → 3.7415 (0.9274; 3.9390) | 5.1543 → 4.0815 (1.0728; 4.3119) | 4.6289 → 3.3139 (1.3151; 3.6387) | [run](https://wandb.ai/leena12/drpt_opus/runs/ack89qbw) |
| GlobalRaw | 4.6689 → 3.8315 (0.8374; 4.0088) | 5.1543 → 4.1946 (0.9597; 4.3978) | 4.6384 → 3.5199 (1.1185; 3.7874) | [run](https://wandb.ai/leena12/drpt_opus/runs/mwvaxlds) |
| LayerwiseRaw | 4.6689 → 3.8010 (0.8679; 3.9910) | 5.1543 → 4.1644 (0.9899; 4.3817) | 4.6415 → 3.5050 (1.1364; 3.7791) | [run](https://wandb.ai/leena12/drpt_opus/runs/nhahb9xj) |
| LayerwiseMuonMatrixSur | 4.6689 → 3.7541 (0.9148; 3.9570) | 5.1543 → 4.1067 (1.0476; 4.3361) | 4.6258 → 3.4179 (1.2078; 3.7085) | [run](https://wandb.ai/leena12/drpt_opus/runs/z8fkc9k3) |
| GlobalSoft | 4.6689 → 3.8032 (0.8657; 3.9879) | 5.1543 → 4.1588 (0.9954; 4.3723) | 4.6396 → 3.4795 (1.1600; 3.7572) | [run](https://wandb.ai/leena12/drpt_opus/runs/s27m9k45) |
| LayerwiseSoft | 4.6689 → 3.7842 (0.8847; 3.9698) | 5.1543 → 4.1318 (1.0225; 4.3510) | 4.6330 → 3.4634 (1.1695; 3.7411) | [run](https://wandb.ai/leena12/drpt_opus/runs/vsfj7e2t) |

## Missing or incomplete requested runs

None.

## W&B panel recipe

In project `leena12/drpt_opus`, filter by group `<campaign>-<setting>-<optimizer>-s42`, where optimizer is `adamw`, `muon`, or legacy `hybrid`, then filter run names to the methods in this report. Use `train/global_step` as the x-axis and add panels for `train/val_loss` (target), `eval/loss` (general held-out), and `train/loss`. Keep evaluation curves unsmoothed; smoothing is useful only for `train/loss`. Exact URLs and groups are in `wandb_runs.csv`.
