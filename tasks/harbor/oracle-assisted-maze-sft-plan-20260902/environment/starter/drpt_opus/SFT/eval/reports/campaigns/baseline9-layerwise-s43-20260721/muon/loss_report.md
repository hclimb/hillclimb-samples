# AdamW/Muon loss-curve comparison

Successful requested runs: **32 / 32**. Missing or incomplete: **0**.

## Reading the metrics

- **Target validation** is `val_loss` on the small target/proxy set used by the selection method. Lower final loss and lower normalized AUC are better, but this set is not independent of selection.
- **General held-out** is `eval_loss` on the larger evaluation set. It is the main loss-based generalization cross-check.
- **Train** compares the mean of the first and last 10% of logged steps; plots use a centered rolling mean. For merged-batch selection runs this is the returned train+target merged loss, while FullTraining returns train-only loss, so cross-method absolute train-loss ranks are not apples-to-apples.
- AUC integrates loss over normalized progress from 0 to 1, making it comparable across run lengths. Lower is better. Reduction is initial minus final, so positive is improvement.
- Both `muon` and legacy `hybrid` runs use the `HybridMuonAdamW` runtime: Muon updates eligible matrix parameters and AdamW updates embeddings, norms, biases, and other ineligible parameters. The `muon` label uses only Muon-managed matrices for the spectral surrogate score; `hybrid` preserves mixed Muon+AdamW scoring.

## Winners by setting

| Setting | Optimizer | Lowest target final | Lowest target AUC | Lowest general final | Lowest general AUC | Largest within-run train reduction |
|---|---|---|---|---|---|---|
| alpaca_samsum | Muon | LayerwiseSoft | LayerwiseSoft | LayerwiseSoft | LayerwiseSoft | LayerwiseMuonOnlySatSur |
| less_squad | Muon | LayerwiseRaw | LayerwiseRaw | LayerwiseRaw | LayerwiseRaw | FullTraining |
| less_tydiqa | Muon | LayerwiseRaw | LayerwiseRaw | LayerwiseSoft | LayerwiseRaw | FullTraining |
| triviaqa_nq | Muon | FullTraining | FullTraining | FullTraining | FullTraining | FullTraining |

## Win counts

Counts are across the four settings; ties count for every tied method. Raw losses are never averaged across different tasks.

| Optimizer | Method | Target final | Target AUC | General final | General AUC | Available settings |
|---|---|---:|---:|---:|---:|---:|
| Muon | FullTraining | 1 | 1 | 1 | 1 | 4 |
| Muon | LayerwiseRaw | 2 | 2 | 1 | 2 | 4 |
| Muon | LayerwiseSoft | 1 | 1 | 2 | 1 | 4 |
| Muon | LayerwiseSoftP | 0 | 0 | 0 | 0 | 4 |
| Muon | LayerwiseMuonMatrixSur | 0 | 0 | 0 | 0 | 4 |
| Muon | LayerwiseMuonOnlyPSur | 0 | 0 | 0 | 0 | 4 |
| Muon | LayerwiseMuonOnlySatSur | 0 | 0 | 0 | 0 | 4 |
| Muon | LayerwiseMuonOnlySatPSur | 0 | 0 | 0 | 0 | 4 |

## Detailed loss reductions

### alpaca_samsum · Muon

| Method | Target initial → final (reduction; AUC) | General initial → final (reduction; AUC) | Train early → late (reduction; AUC) | W&B |
|---|---:|---:|---:|---|
| FullTraining | 2.2578 → 2.0576 (0.2003; 2.0815) | 2.0594 → 1.9471 (0.1123; 1.9600) | 1.7145 → 1.6696 (0.0449; 1.6470) | [run](https://wandb.ai/leena12/drpt_opus/runs/xsz8m6y8) |
| LayerwiseRaw | 2.2578 → 2.0365 (0.2213; 2.0671) | 2.0594 → 1.9383 (0.1211; 1.9538) | 1.7348 → 1.6948 (0.0400; 1.6723) | [run](https://wandb.ai/leena12/drpt_opus/runs/6sgf2yn3) |
| LayerwiseSoft | 2.2578 → 2.0269 (0.2309; 2.0604) | 2.0594 → 1.9356 (0.1238; 1.9512) | 1.7343 → 1.6925 (0.0418; 1.6704) | [run](https://wandb.ai/leena12/drpt_opus/runs/rkpc9jcz) |
| LayerwiseSoftP | 2.2578 → 2.0598 (0.1981; 2.0913) | 2.0594 → 1.9509 (0.1085; 1.9669) | 1.7424 → 1.7172 (0.0252; 1.6928) | [run](https://wandb.ai/leena12/drpt_opus/runs/0i73eqnc) |
| LayerwiseMuonMatrixSur | 2.2578 → 2.0574 (0.2004; 2.0821) | 2.0594 → 1.9479 (0.1115; 1.9610) | 1.7316 → 1.6860 (0.0456; 1.6644) | [run](https://wandb.ai/leena12/drpt_opus/runs/45c2nuqf) |
| LayerwiseMuonOnlyPSur | 2.2578 → 2.0561 (0.2018; 2.0801) | 2.0594 → 1.9463 (0.1131; 1.9599) | 1.7315 → 1.6858 (0.0457; 1.6643) | [run](https://wandb.ai/leena12/drpt_opus/runs/n74hb80v) |
| LayerwiseMuonOnlySatSur | 2.2578 → 2.0593 (0.1986; 2.0812) | 2.0594 → 1.9469 (0.1125; 1.9606) | 1.7313 → 1.6855 (0.0458; 1.6638) | [run](https://wandb.ai/leena12/drpt_opus/runs/6orxiqeu) |
| LayerwiseMuonOnlySatPSur | 2.2578 → 2.0559 (0.2020; 2.0810) | 2.0594 → 1.9469 (0.1125; 1.9604) | 1.7316 → 1.6865 (0.0450; 1.6647) | [run](https://wandb.ai/leena12/drpt_opus/runs/w91dkgbk) |

### less_squad · Muon

| Method | Target initial → final (reduction; AUC) | General initial → final (reduction; AUC) | Train early → late (reduction; AUC) | W&B |
|---|---:|---:|---:|---|
| FullTraining | 4.6236 → 4.3379 (0.2857; 4.3975) | 4.8678 → 4.5752 (0.2927; 4.6299) | 1.9751 → 1.8831 (0.0920; 1.9058) | [run](https://wandb.ai/leena12/drpt_opus/runs/8mor3qdp) |
| LayerwiseRaw | 4.6236 → 4.3208 (0.3028; 4.3818) | 4.8678 → 4.5485 (0.3193; 4.6110) | 2.0099 → 1.9270 (0.0828; 1.9481) | [run](https://wandb.ai/leena12/drpt_opus/runs/f45dxets) |
| LayerwiseSoft | 4.6236 → 4.3339 (0.2897; 4.3947) | 4.8678 → 4.5647 (0.3031; 4.6248) | 2.0089 → 1.9252 (0.0837; 1.9467) | [run](https://wandb.ai/leena12/drpt_opus/runs/e29wdm80) |
| LayerwiseSoftP | 4.6236 → 4.3750 (0.2486; 4.4298) | 4.8678 → 4.6208 (0.2471; 4.6710) | 2.0147 → 1.9529 (0.0617; 1.9701) | [run](https://wandb.ai/leena12/drpt_opus/runs/arqoc94l) |
| LayerwiseMuonMatrixSur | 4.6236 → 4.3372 (0.2864; 4.3947) | 4.8678 → 4.5707 (0.2971; 4.6267) | 2.0060 → 1.9157 (0.0903; 1.9380) | [run](https://wandb.ai/leena12/drpt_opus/runs/fdpol1o2) |
| LayerwiseMuonOnlyPSur | 4.6236 → 4.3342 (0.2894; 4.3922) | 4.8678 → 4.5693 (0.2986; 4.6231) | 2.0059 → 1.9153 (0.0907; 1.9377) | [run](https://wandb.ai/leena12/drpt_opus/runs/200kiah4) |
| LayerwiseMuonOnlySatSur | 4.6236 → 4.3395 (0.2841; 4.3945) | 4.8678 → 4.5678 (0.3001; 4.6252) | 2.0063 → 1.9155 (0.0907; 1.9380) | [run](https://wandb.ai/leena12/drpt_opus/runs/n41150wp) |
| LayerwiseMuonOnlySatPSur | 4.6236 → 4.3344 (0.2892; 4.3892) | 4.8678 → 4.5632 (0.3047; 4.6203) | 2.0061 → 1.9148 (0.0912; 1.9375) | [run](https://wandb.ai/leena12/drpt_opus/runs/frnglhk7) |

### less_tydiqa · Muon

| Method | Target initial → final (reduction; AUC) | General initial → final (reduction; AUC) | Train early → late (reduction; AUC) | W&B |
|---|---:|---:|---:|---|
| FullTraining | 1.3386 → 1.2338 (0.1048; 1.2575) | 1.9630 → 1.8146 (0.1484; 1.8454) | 1.9749 → 1.8833 (0.0916; 1.9058) | [run](https://wandb.ai/leena12/drpt_opus/runs/rrycjlwl) |
| LayerwiseRaw | 1.3386 → 1.2247 (0.1139; 1.2452) | 1.9630 → 1.7995 (0.1635; 1.8292) | 1.9580 → 1.8763 (0.0818; 1.8975) | [run](https://wandb.ai/leena12/drpt_opus/runs/jamtconm) |
| LayerwiseSoft | 1.3386 → 1.2264 (0.1122; 1.2476) | 1.9630 → 1.7984 (0.1646; 1.8319) | 1.9579 → 1.8761 (0.0819; 1.8975) | [run](https://wandb.ai/leena12/drpt_opus/runs/7f5g7qn2) |
| LayerwiseSoftP | 1.3386 → 1.2428 (0.0957; 1.2621) | 1.9630 → 1.8203 (0.1427; 1.8522) | 1.9631 → 1.9017 (0.0615; 1.9193) | [run](https://wandb.ai/leena12/drpt_opus/runs/jbml6cv2) |
| LayerwiseMuonMatrixSur | 1.3386 → 1.2315 (0.1071; 1.2510) | 1.9630 → 1.8077 (0.1553; 1.8380) | 1.9548 → 1.8657 (0.0891; 1.8883) | [run](https://wandb.ai/leena12/drpt_opus/runs/m2xfbf68) |
| LayerwiseMuonOnlyPSur | 1.3386 → 1.2282 (0.1103; 1.2495) | 1.9630 → 1.8053 (0.1577; 1.8364) | 1.9546 → 1.8655 (0.0891; 1.8882) | [run](https://wandb.ai/leena12/drpt_opus/runs/6kpmr541) |
| LayerwiseMuonOnlySatSur | 1.3386 → 1.2298 (0.1088; 1.2511) | 1.9630 → 1.8095 (0.1535; 1.8386) | 1.9548 → 1.8661 (0.0887; 1.8885) | [run](https://wandb.ai/leena12/drpt_opus/runs/9vhiyyp5) |
| LayerwiseMuonOnlySatPSur | 1.3386 → 1.2289 (0.1097; 1.2502) | 1.9630 → 1.8050 (0.1580; 1.8366) | 1.9548 → 1.8655 (0.0893; 1.8882) | [run](https://wandb.ai/leena12/drpt_opus/runs/la2dk5fy) |

### triviaqa_nq · Muon

| Method | Target initial → final (reduction; AUC) | General initial → final (reduction; AUC) | Train early → late (reduction; AUC) | W&B |
|---|---:|---:|---:|---|
| FullTraining | 4.1855 → 3.2406 (0.9449; 3.4508) | 5.1573 → 4.0768 (1.0805; 4.3116) | 4.5676 → 3.3757 (1.1919; 3.6800) | [run](https://wandb.ai/leena12/drpt_opus/runs/vgpz6lla) |
| LayerwiseRaw | 4.1855 → 3.3254 (0.8601; 3.5085) | 5.1573 → 4.1692 (0.9881; 4.3822) | 4.5003 → 3.4862 (1.0141; 3.7480) | [run](https://wandb.ai/leena12/drpt_opus/runs/snoik3g1) |
| LayerwiseSoft | 4.1855 → 3.2746 (0.9109; 3.4765) | 5.1573 → 4.1276 (1.0297; 4.3477) | 4.4958 → 3.4278 (1.0680; 3.7013) | [run](https://wandb.ai/leena12/drpt_opus/runs/xc234k6o) |
| LayerwiseSoftP | 4.1855 → 3.5248 (0.6607; 3.6686) | 5.1573 → 4.4250 (0.7324; 4.5849) | 4.5497 → 3.8082 (0.7415; 4.0053) | [run](https://wandb.ai/leena12/drpt_opus/runs/ldw7pxqq) |
| LayerwiseMuonMatrixSur | 4.1855 → 3.2782 (0.9074; 3.4748) | 5.1573 → 4.1149 (1.0425; 4.3408) | 4.4896 → 3.3958 (1.0938; 3.6744) | [run](https://wandb.ai/leena12/drpt_opus/runs/ydefmwea) |
| LayerwiseMuonOnlyPSur | 4.1855 → 3.2712 (0.9143; 3.4701) | 5.1573 → 4.1071 (1.0503; 4.3335) | 4.4906 → 3.3881 (1.1025; 3.6680) | [run](https://wandb.ai/leena12/drpt_opus/runs/jasgx5dy) |
| LayerwiseMuonOnlySatSur | 4.1855 → 3.2686 (0.9170; 3.4754) | 5.1573 → 4.1111 (1.0462; 4.3412) | 4.4898 → 3.3934 (1.0964; 3.6751) | [run](https://wandb.ai/leena12/drpt_opus/runs/y9b8pfrs) |
| LayerwiseMuonOnlySatPSur | 4.1855 → 3.2754 (0.9102; 3.4726) | 5.1573 → 4.1124 (1.0449; 4.3381) | 4.4891 → 3.3908 (1.0983; 3.6722) | [run](https://wandb.ai/leena12/drpt_opus/runs/577ug9qz) |

## Missing or incomplete requested runs

None.

## W&B panel recipe

In project `leena12/drpt_opus`, filter by group `<campaign>-<setting>-<optimizer>-s42`, where optimizer is `adamw`, `muon`, or legacy `hybrid`, then filter run names to the methods in this report. Use `train/global_step` as the x-axis and add panels for `train/val_loss` (target), `eval/loss` (general held-out), and `train/loss`. Keep evaluation curves unsmoothed; smoothing is useful only for `train/loss`. Exact URLs and groups are in `wandb_runs.csv`.
