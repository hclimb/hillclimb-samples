# AdamW/Muon loss-curve comparison

Successful requested runs: **28 / 28**. Missing or incomplete: **0**.

## Reading the metrics

- **Target validation** is `val_loss` on the small target/proxy set used by the selection method. Lower final loss and lower normalized AUC are better, but this set is not independent of selection.
- **General held-out** is `eval_loss` on the larger evaluation set. It is the main loss-based generalization cross-check.
- **Train** compares the mean of the first and last 10% of logged steps; plots use a centered rolling mean. For merged-batch selection runs this is the returned train+target merged loss, while FullTraining returns train-only loss, so cross-method absolute train-loss ranks are not apples-to-apples.
- AUC integrates loss over normalized progress from 0 to 1, making it comparable across run lengths. Lower is better. Reduction is initial minus final, so positive is improvement.
- Both `muon` and legacy `hybrid` runs use the `HybridMuonAdamW` runtime: Muon updates eligible matrix parameters and AdamW updates embeddings, norms, biases, and other ineligible parameters. The `muon` label uses only Muon-managed matrices for the spectral surrogate score; `hybrid` preserves mixed Muon+AdamW scoring.

## Winners by setting

| Setting | Optimizer | Lowest target final | Lowest target AUC | Lowest general final | Lowest general AUC | Largest within-run train reduction |
|---|---|---|---|---|---|---|
| alpaca_samsum | AdamW | LayerwiseSoft | LayerwiseSoft | LayerwiseRaw | LayerwiseSoft | LayerwiseOptA |
| less_squad | AdamW | LayerwiseOptA | LayerwiseOptA | LayerwiseOptA | LayerwiseOptA | FullTraining |
| less_tydiqa | AdamW | LayerwiseOptA | LayerwiseOptA | LayerwiseOptA | LayerwiseOptA | FullTraining |
| triviaqa_nq | AdamW | LayerwiseRaw | LayerwiseRaw | LayerwiseRaw | LayerwiseRaw | LayerwiseRaw |

## Win counts

Counts are across the four settings; ties count for every tied method. Raw losses are never averaged across different tasks.

| Optimizer | Method | Target final | Target AUC | General final | General AUC | Available settings |
|---|---|---:|---:|---:|---:|---:|
| AdamW | FullTraining | 0 | 0 | 0 | 0 | 4 |
| AdamW | GlobalRaw | 0 | 0 | 0 | 0 | 4 |
| AdamW | LayerwiseRaw | 1 | 1 | 2 | 1 | 4 |
| AdamW | GlobalOptA | 0 | 0 | 0 | 0 | 4 |
| AdamW | LayerwiseOptA | 2 | 2 | 2 | 2 | 4 |
| AdamW | GlobalSoft | 0 | 0 | 0 | 0 | 4 |
| AdamW | LayerwiseSoft | 1 | 1 | 0 | 1 | 4 |

## Detailed loss reductions

### alpaca_samsum · AdamW

| Method | Target initial → final (reduction; AUC) | General initial → final (reduction; AUC) | Train early → late (reduction; AUC) | W&B |
|---|---:|---:|---:|---|
| FullTraining | 2.2138 → 1.6159 (0.5978; 1.6170) | 2.0607 → 1.7565 (0.3042; 1.7610) | 1.4686 → 1.3766 (0.0920; 1.3921) | [run](https://wandb.ai/leena12/drpt_opus/runs/vdbi1q2u) |
| GlobalRaw | 2.2138 → 1.4211 (0.7927; 1.4484) | 2.0607 → 1.6623 (0.3984; 1.6796) | 1.4908 → 1.3934 (0.0973; 1.4097) | [run](https://wandb.ai/leena12/drpt_opus/runs/mpynkth4) |
| LayerwiseRaw | 2.2138 → 1.0803 (1.1334; 1.1593) | 2.0607 → 1.5730 (0.4877; 1.5957) | 1.4884 → 1.3848 (0.1037; 1.4012) | [run](https://wandb.ai/leena12/drpt_opus/runs/hiucwmdn) |
| GlobalOptA | 2.2138 → 1.3339 (0.8798; 1.3730) | 2.0607 → 1.6236 (0.4370; 1.6447) | 1.4923 → 1.3914 (0.1009; 1.4088) | [run](https://wandb.ai/leena12/drpt_opus/runs/zaubzys9) |
| LayerwiseOptA | 2.2138 → 1.0718 (1.1420; 1.1482) | 2.0607 → 1.5771 (0.4835; 1.5975) | 1.4882 → 1.3838 (0.1044; 1.4008) | [run](https://wandb.ai/leena12/drpt_opus/runs/bru2az8t) |
| GlobalSoft | 2.2138 → 1.3418 (0.8720; 1.3836) | 2.0607 → 1.6353 (0.4254; 1.6565) | 1.4856 → 1.3869 (0.0986; 1.4041) | [run](https://wandb.ai/leena12/drpt_opus/runs/ofbirn2f) |
| LayerwiseSoft | 2.2138 → 1.0676 (1.1462; 1.1406) | 2.0607 → 1.5752 (0.4855; 1.5946) | 1.4832 → 1.3803 (0.1029; 1.3969) | [run](https://wandb.ai/leena12/drpt_opus/runs/1v2pv132) |

### less_squad · AdamW

| Method | Target initial → final (reduction; AUC) | General initial → final (reduction; AUC) | Train early → late (reduction; AUC) | W&B |
|---|---:|---:|---:|---|
| FullTraining | 4.1166 → 3.9027 (0.2139; 3.8773) | 4.8667 → 4.5702 (0.2965; 4.5489) | 1.7458 → 1.4927 (0.2531; 1.5404) | [run](https://wandb.ai/leena12/drpt_opus/runs/t75r2vh7) |
| GlobalRaw | 4.1166 → 3.2346 (0.8820; 3.2933) | 4.8667 → 3.9171 (0.9496; 3.9763) | 1.8047 → 1.5635 (0.2412; 1.6106) | [run](https://wandb.ai/leena12/drpt_opus/runs/10gusn7w) |
| LayerwiseRaw | 4.1166 → 2.7963 (1.3203; 2.9317) | 4.8667 → 3.5322 (1.3345; 3.6536) | 1.8019 → 1.5584 (0.2435; 1.6052) | [run](https://wandb.ai/leena12/drpt_opus/runs/ixquxado) |
| GlobalOptA | 4.1166 → 3.1931 (0.9235; 3.2300) | 4.8667 → 3.8612 (1.0055; 3.8993) | 1.8095 → 1.5693 (0.2402; 1.6165) | [run](https://wandb.ai/leena12/drpt_opus/runs/5cy10iqk) |
| LayerwiseOptA | 4.1166 → 2.6830 (1.4336; 2.8117) | 4.8667 → 3.3901 (1.4766; 3.5009) | 1.8041 → 1.5584 (0.2456; 1.6050) | [run](https://wandb.ai/leena12/drpt_opus/runs/52n7yh31) |
| GlobalSoft | 4.1166 → 3.1782 (0.9384; 3.2227) | 4.8667 → 3.8101 (1.0565; 3.8662) | 1.7986 → 1.5521 (0.2465; 1.5999) | [run](https://wandb.ai/leena12/drpt_opus/runs/qx27tawf) |
| LayerwiseSoft | 4.1166 → 2.7651 (1.3515; 2.8719) | 4.8667 → 3.4396 (1.4270; 3.5527) | 1.7953 → 1.5475 (0.2478; 1.5942) | [run](https://wandb.ai/leena12/drpt_opus/runs/n5mdi1dt) |

### less_tydiqa · AdamW

| Method | Target initial → final (reduction; AUC) | General initial → final (reduction; AUC) | Train early → late (reduction; AUC) | W&B |
|---|---:|---:|---:|---|
| FullTraining | 1.4262 → 1.1553 (0.2708; 1.1468) | 1.9643 → 1.6052 (0.3591; 1.5802) | 1.7458 → 1.4926 (0.2532; 1.5403) | [run](https://wandb.ai/leena12/drpt_opus/runs/ls5htcc2) |
| GlobalRaw | 1.4262 → 0.9522 (0.4740; 0.9640) | 1.9643 → 1.2103 (0.7540; 1.2237) | 1.7499 → 1.5119 (0.2380; 1.5561) | [run](https://wandb.ai/leena12/drpt_opus/runs/wq7zmmgx) |
| LayerwiseRaw | 1.4262 → 0.7444 (0.6817; 0.7888) | 1.9643 → 0.9732 (0.9911; 1.0121) | 1.7458 → 1.5053 (0.2404; 1.5497) | [run](https://wandb.ai/leena12/drpt_opus/runs/x3z1tj8z) |
| GlobalOptA | 1.4262 → 0.9383 (0.4878; 0.9366) | 1.9643 → 1.2593 (0.7050; 1.2293) | 1.7535 → 1.5125 (0.2410; 1.5580) | [run](https://wandb.ai/leena12/drpt_opus/runs/157qzc8e) |
| LayerwiseOptA | 1.4262 → 0.6504 (0.7758; 0.7154) | 1.9643 → 0.8036 (1.1607; 0.8892) | 1.7466 → 1.5020 (0.2446; 1.5482) | [run](https://wandb.ai/leena12/drpt_opus/runs/gggtwwj6) |
| GlobalSoft | 1.4262 → 0.9204 (0.5058; 0.9394) | 1.9643 → 1.2311 (0.7332; 1.2243) | 1.7401 → 1.4964 (0.2436; 1.5422) | [run](https://wandb.ai/leena12/drpt_opus/runs/mpi59uzg) |
| LayerwiseSoft | 1.4262 → 0.6537 (0.7724; 0.7180) | 1.9643 → 0.8245 (1.1398; 0.8984) | 1.7388 → 1.4928 (0.2459; 1.5385) | [run](https://wandb.ai/leena12/drpt_opus/runs/p8f8h7v5) |

### triviaqa_nq · AdamW

| Method | Target initial → final (reduction; AUC) | General initial → final (reduction; AUC) | Train early → late (reduction; AUC) | W&B |
|---|---:|---:|---:|---|
| FullTraining | 4.6689 → 2.5329 (2.1360; 2.5666) | 5.1543 → 2.6042 (2.5500; 2.6271) | 2.0324 → 1.2922 (0.7402; 1.4473) | [run](https://wandb.ai/leena12/drpt_opus/runs/q2sgubuk) |
| GlobalRaw | 4.6689 → 2.3593 (2.3096; 2.4345) | 5.1543 → 2.5140 (2.6402; 2.5656) | 2.2196 → 1.5056 (0.7141; 1.6458) | [run](https://wandb.ai/leena12/drpt_opus/runs/r8w5dr0x) |
| LayerwiseRaw | 4.6689 → 2.1071 (2.5618; 2.2160) | 5.1543 → 2.4711 (2.6831; 2.5196) | 2.1961 → 1.4520 (0.7440; 1.5964) | [run](https://wandb.ai/leena12/drpt_opus/runs/qjwehlk2) |
| GlobalOptA | 4.6689 → 2.3381 (2.3309; 2.4203) | 5.1543 → 2.5015 (2.6527; 2.5487) | 2.2228 → 1.4998 (0.7229; 1.6363) | [run](https://wandb.ai/leena12/drpt_opus/runs/n64qvf1b) |
| LayerwiseOptA | 4.6689 → 2.1193 (2.5496; 2.2164) | 5.1543 → 2.4760 (2.6783; 2.5225) | 2.1978 → 1.4557 (0.7421; 1.5977) | [run](https://wandb.ai/leena12/drpt_opus/runs/z0zbp0ca) |
| GlobalSoft | 4.6689 → 2.3418 (2.3271; 2.4169) | 5.1543 → 2.5050 (2.6493; 2.5494) | 2.2023 → 1.4876 (0.7147; 1.6244) | [run](https://wandb.ai/leena12/drpt_opus/runs/iy8xdnwz) |
| LayerwiseSoft | 4.6689 → 2.1390 (2.5299; 2.2369) | 5.1543 → 2.4873 (2.6669; 2.5286) | 2.1779 → 1.4526 (0.7253; 1.5904) | [run](https://wandb.ai/leena12/drpt_opus/runs/h88fw854) |

## Missing or incomplete requested runs

None.

## W&B panel recipe

In project `leena12/drpt_opus`, filter by group `<campaign>-<setting>-<optimizer>-s42`, where optimizer is `adamw`, `muon`, or legacy `hybrid`, then filter run names to the methods in this report. Use `train/global_step` as the x-axis and add panels for `train/val_loss` (target), `eval/loss` (general held-out), and `train/loss`. Keep evaluation curves unsmoothed; smoothing is useful only for `train/loss`. Exact URLs and groups are in `wandb_runs.csv`.
