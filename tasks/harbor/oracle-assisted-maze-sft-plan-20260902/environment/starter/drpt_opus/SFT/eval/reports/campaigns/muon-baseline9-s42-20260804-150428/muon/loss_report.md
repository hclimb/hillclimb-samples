# AdamW/Muon loss-curve comparison

Successful requested runs: **32 / 32**. Missing or incomplete: **0**.

## Reading the metrics

- **Target validation** is `val_loss` on the small target/proxy set used by the selection method. Lower final loss and lower normalized AUC are better, but this set is not independent of selection.
- **General held-out** is `eval_loss` on the larger evaluation set. It is the main loss-based generalization cross-check.
- **Train** compares the mean of the first and last 10% of logged steps; plots use a centered rolling mean. For merged-batch selection runs this is the returned train+target merged loss, while FullTraining returns train-only loss, so cross-method absolute train-loss ranks are not apples-to-apples.
- AUC integrates loss over normalized progress from 0 to 1, making it comparable across run lengths. Lower is better. Reduction is initial minus final, so positive is improvement.
- Both `muon` and legacy `hybrid` runs prioritize official `torch.optim.Muon` for eligible matrices and use auxiliary AdamW for embeddings, norms, biases, heads, and other ineligible parameters. The local Muon is used only if the official backend is unavailable or incompatible. The `muon` label uses only Muon-managed matrices for the spectral surrogate score; `hybrid` preserves mixed Muon+AdamW scoring.

## Winners by setting

| Setting | Optimizer | Lowest target final | Lowest target AUC | Lowest general final | Lowest general AUC | Largest within-run train reduction |
|---|---|---|---|---|---|---|
| alpaca_samsum | Muon | LayerwiseSoft | LayerwiseSoft | LayerwiseSoft | LayerwiseSoft | LayerwiseMuonSur |
| less_squad | Muon | LayerwiseRaw | LayerwiseSoft | LayerwiseRaw | LayerwiseSoft | FullTraining |
| less_tydiqa | Muon | LayerwiseRaw | LayerwiseRaw | LayerwiseRaw | LayerwiseRaw | FullTraining |
| triviaqa_nq | Muon | FullTraining | FullTraining | FullTraining | FullTraining | FullTraining |

## Win counts

Counts are across the four settings; ties count for every tied method. Raw losses are never averaged across different tasks.

| Optimizer | Method | Target final | Target AUC | General final | General AUC | Available settings |
|---|---|---:|---:|---:|---:|---:|
| Muon | FullTraining | 1 | 1 | 1 | 1 | 4 |
| Muon | LayerwiseRaw | 2 | 1 | 2 | 1 | 4 |
| Muon | LayerwiseSoft | 1 | 2 | 1 | 2 | 4 |
| Muon | LayerwiseSoftP | 0 | 0 | 0 | 0 | 4 |
| Muon | LayerwiseMuonSur | 0 | 0 | 0 | 0 | 4 |
| Muon | LayerwiseMuonPSur | 0 | 0 | 0 | 0 | 4 |
| Muon | LayerwiseMuonSatSur | 0 | 0 | 0 | 0 | 4 |
| Muon | LayerwiseMuonSatPSur | 0 | 0 | 0 | 0 | 4 |

## Detailed loss reductions

### alpaca_samsum · Muon

| Method | Target initial → final (reduction; AUC) | General initial → final (reduction; AUC) | Train early → late (reduction; AUC) | W&B |
|---|---:|---:|---:|---|
| FullTraining | 2.2209 → 2.0094 (0.2115; 2.0362) | 2.0597 → 1.9494 (0.1103; 1.9622) | 1.7045 → 1.6370 (0.0675; 1.6480) | [run](https://wandb.ai/leena12/drpt_opus/runs/9uw748a9) |
| LayerwiseRaw | 2.2209 → 1.9949 (0.2261; 2.0234) | 2.0597 → 1.9381 (0.1217; 1.9531) | 1.7245 → 1.6612 (0.0633; 1.6721) | [run](https://wandb.ai/leena12/drpt_opus/runs/0qfxin0w) |
| LayerwiseSoft | 2.2209 → 1.9915 (0.2294; 2.0193) | 2.0597 → 1.9373 (0.1225; 1.9521) | 1.7233 → 1.6606 (0.0627; 1.6711) | [run](https://wandb.ai/leena12/drpt_opus/runs/u5gcnwkt) |
| LayerwiseSoftP | 2.2209 → 2.0109 (0.2101; 2.0425) | 2.0597 → 1.9463 (0.1135; 1.9632) | 1.7311 → 1.6822 (0.0489; 1.6915) | [run](https://wandb.ai/leena12/drpt_opus/runs/m8qmh1lk) |
| LayerwiseMuonSur | 2.2209 → 2.0078 (0.2132; 2.0350) | 2.0597 → 1.9484 (0.1113; 1.9609) | 1.7203 → 1.6520 (0.0683; 1.6632) | [run](https://wandb.ai/leena12/drpt_opus/runs/v525y1pj) |
| LayerwiseMuonPSur | 2.2209 → 2.0063 (0.2147; 2.0340) | 2.0597 → 1.9476 (0.1121; 1.9609) | 1.7201 → 1.6522 (0.0679; 1.6634) | [run](https://wandb.ai/leena12/drpt_opus/runs/mfsra2k0) |
| LayerwiseMuonSatSur | 2.2209 → 2.0105 (0.2104; 2.0352) | 2.0597 → 1.9484 (0.1114; 1.9614) | 1.7203 → 1.6521 (0.0682; 1.6634) | [run](https://wandb.ai/leena12/drpt_opus/runs/o248xrfb) |
| LayerwiseMuonSatPSur | 2.2209 → 2.0078 (0.2131; 2.0347) | 2.0597 → 1.9483 (0.1115; 1.9612) | 1.7204 → 1.6525 (0.0679; 1.6635) | [run](https://wandb.ai/leena12/drpt_opus/runs/f64o7umg) |

### less_squad · Muon

| Method | Target initial → final (reduction; AUC) | General initial → final (reduction; AUC) | Train early → late (reduction; AUC) | W&B |
|---|---:|---:|---:|---|
| FullTraining | 4.1101 → 3.8270 (0.2832; 3.8772) | 4.8642 → 4.5757 (0.2885; 4.6293) | 1.9664 → 1.8586 (0.1078; 1.8844) | [run](https://wandb.ai/leena12/drpt_opus/runs/ad3wsvd2) |
| LayerwiseRaw | 4.1101 → 3.7879 (0.3223; 3.8527) | 4.8642 → 4.5366 (0.3276; 4.6050) | 1.9980 → 1.8989 (0.0992; 1.9231) | [run](https://wandb.ai/leena12/drpt_opus/runs/1uey489d) |
| LayerwiseSoft | 4.1101 → 3.7964 (0.3138; 3.8504) | 4.8642 → 4.5435 (0.3207; 4.6044) | 1.9978 → 1.8986 (0.0993; 1.9229) | [run](https://wandb.ai/leena12/drpt_opus/runs/dwchua3y) |
| LayerwiseSoftP | 4.1101 → 3.8565 (0.2537; 3.9078) | 4.8642 → 4.6102 (0.2540; 4.6613) | 2.0034 → 1.9252 (0.0782; 1.9450) | [run](https://wandb.ai/leena12/drpt_opus/runs/3auyg2i8) |
| LayerwiseMuonSur | 4.1101 → 3.8074 (0.3028; 3.8664) | 4.8642 → 4.5589 (0.3052; 4.6167) | 1.9951 → 1.8882 (0.1069; 1.9137) | [run](https://wandb.ai/leena12/drpt_opus/runs/tlxmrhpt) |
| LayerwiseMuonPSur | 4.1101 → 3.8173 (0.2929; 3.8660) | 4.8642 → 4.5591 (0.3051; 4.6166) | 1.9954 → 1.8884 (0.1070; 1.9138) | [run](https://wandb.ai/leena12/drpt_opus/runs/ii2e1s6l) |
| LayerwiseMuonSatSur | 4.1101 → 3.8144 (0.2957; 3.8696) | 4.8642 → 4.5604 (0.3038; 4.6192) | 1.9951 → 1.8884 (0.1068; 1.9138) | [run](https://wandb.ai/leena12/drpt_opus/runs/pg2fhnbt) |
| LayerwiseMuonSatPSur | 4.1101 → 3.8063 (0.3038; 3.8645) | 4.8642 → 4.5572 (0.3070; 4.6161) | 1.9950 → 1.8885 (0.1065; 1.9136) | [run](https://wandb.ai/leena12/drpt_opus/runs/1clm6118) |

### less_tydiqa · Muon

| Method | Target initial → final (reduction; AUC) | General initial → final (reduction; AUC) | Train early → late (reduction; AUC) | W&B |
|---|---:|---:|---:|---|
| FullTraining | 1.4223 → 1.3260 (0.0964; 1.3435) | 1.9633 → 1.8184 (0.1449; 1.8447) | 1.9664 → 1.8585 (0.1079; 1.8843) | [run](https://wandb.ai/leena12/drpt_opus/runs/sb8n1l8i) |
| LayerwiseRaw | 1.4223 → 1.3079 (0.1145; 1.3302) | 1.9633 → 1.7941 (0.1692; 1.8271) | 1.9404 → 1.8400 (0.1004; 1.8632) | [run](https://wandb.ai/leena12/drpt_opus/runs/czsj6stz) |
| LayerwiseSoft | 1.4223 → 1.3143 (0.1081; 1.3333) | 1.9633 → 1.7987 (0.1645; 1.8310) | 1.9404 → 1.8395 (0.1009; 1.8631) | [run](https://wandb.ai/leena12/drpt_opus/runs/5cocvv3d) |
| LayerwiseSoftP | 1.4223 → 1.3174 (0.1049; 1.3441) | 1.9633 → 1.8115 (0.1518; 1.8466) | 1.9464 → 1.8670 (0.0794; 1.8860) | [run](https://wandb.ai/leena12/drpt_opus/runs/vzbc0vwl) |
| LayerwiseMuonSur | 1.4223 → 1.3211 (0.1013; 1.3389) | 1.9633 → 1.8095 (0.1537; 1.8379) | 1.9375 → 1.8300 (0.1075; 1.8548) | [run](https://wandb.ai/leena12/drpt_opus/runs/xipi2qmm) |
| LayerwiseMuonPSur | 1.4223 → 1.3185 (0.1038; 1.3375) | 1.9633 → 1.8080 (0.1552; 1.8370) | 1.9380 → 1.8301 (0.1079; 1.8548) | [run](https://wandb.ai/leena12/drpt_opus/runs/1l9ykzqs) |
| LayerwiseMuonSatSur | 1.4223 → 1.3193 (0.1030; 1.3385) | 1.9633 → 1.8087 (0.1546; 1.8380) | 1.9376 → 1.8299 (0.1077; 1.8547) | [run](https://wandb.ai/leena12/drpt_opus/runs/dqcn8y4y) |
| LayerwiseMuonSatPSur | 1.4223 → 1.3192 (0.1031; 1.3382) | 1.9633 → 1.8046 (0.1587; 1.8373) | 1.9377 → 1.8304 (0.1074; 1.8550) | [run](https://wandb.ai/leena12/drpt_opus/runs/g0iqmibl) |

### triviaqa_nq · Muon

| Method | Target initial → final (reduction; AUC) | General initial → final (reduction; AUC) | Train early → late (reduction; AUC) | W&B |
|---|---:|---:|---:|---|
| FullTraining | 4.6699 → 3.7509 (0.9190; 3.9465) | 5.1503 → 4.0900 (1.0603; 4.3220) | 4.6299 → 3.3339 (1.2959; 3.6522) | [run](https://wandb.ai/leena12/drpt_opus/runs/hbj5gmkc) |
| LayerwiseRaw | 4.6699 → 3.8116 (0.8582; 3.9976) | 5.1503 → 4.1795 (0.9709; 4.3914) | 4.6431 → 3.5243 (1.1187; 3.7913) | [run](https://wandb.ai/leena12/drpt_opus/runs/3elcqod9) |
| LayerwiseSoft | 4.6699 → 3.7688 (0.9010; 3.9762) | 5.1503 → 4.1368 (1.0136; 4.3635) | 4.6359 → 3.4905 (1.1454; 3.7605) | [run](https://wandb.ai/leena12/drpt_opus/runs/laxi12vk) |
| LayerwiseSoftP | 4.6699 → 4.0237 (0.6461; 4.1676) | 5.1503 → 4.4189 (0.7315; 4.5795) | 4.6835 → 3.8468 (0.8367; 4.0415) | [run](https://wandb.ai/leena12/drpt_opus/runs/qduzl7fq) |
| LayerwiseMuonSur | 4.6699 → 3.7587 (0.9112; 3.9569) | 5.1503 → 4.1181 (1.0322; 4.3403) | 4.6286 → 3.4294 (1.1992; 3.7172) | [run](https://wandb.ai/leena12/drpt_opus/runs/rxxib1zf) |
| LayerwiseMuonPSur | 4.6699 → 3.7660 (0.9039; 3.9580) | 5.1503 → 4.1241 (1.0263; 4.3401) | 4.6294 → 3.4244 (1.2050; 3.7166) | [run](https://wandb.ai/leena12/drpt_opus/runs/cy4l5cki) |
| LayerwiseMuonSatSur | 4.6699 → 3.7728 (0.8971; 3.9644) | 5.1503 → 4.1197 (1.0307; 4.3470) | 4.6282 → 3.4424 (1.1858; 3.7258) | [run](https://wandb.ai/leena12/drpt_opus/runs/rhec4hr7) |
| LayerwiseMuonSatPSur | 4.6699 → 3.7575 (0.9124; 3.9609) | 5.1503 → 4.1246 (1.0258; 4.3439) | 4.6302 → 3.4348 (1.1953; 3.7209) | [run](https://wandb.ai/leena12/drpt_opus/runs/ofiow1bg) |

## Missing or incomplete requested runs

None.

## W&B panel recipe

In project `leena12/drpt_opus`, filter by group `<campaign>-<setting>-<optimizer>-s42`, where optimizer is `adamw`, `muon`, or legacy `hybrid`, then filter run names to the methods in this report. Use `train/global_step` as the x-axis and add panels for `train/val_loss` (target), `eval/loss` (general held-out), and `train/loss`. Keep evaluation curves unsmoothed; smoothing is useful only for `train/loss`. Exact URLs and groups are in `wandb_runs.csv`.
