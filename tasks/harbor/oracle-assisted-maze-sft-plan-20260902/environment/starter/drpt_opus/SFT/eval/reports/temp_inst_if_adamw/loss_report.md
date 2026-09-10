# AdamW/Muon loss-curve comparison

Successful requested runs: **25 / 25**. Missing or incomplete: **0**.

## Reading the metrics

- **Target validation** is `target_val_loss` on the disjoint 128-example target monitoring split. It is never used for selection; lower final loss and lower normalized AUC are better.
- **General held-out** is `general_val_loss` on the disjoint 512-example general validation split. It is the main loss-based generalization cross-check.
- **Train** compares the mean of the first and last 10% of logged steps; plots use a centered rolling mean. For merged-batch selection runs this is the returned train+target merged loss, while FullTraining returns train-only loss, so cross-method absolute train-loss ranks are not apples-to-apples.
- AUC integrates loss over normalized progress from 0 to 1, making it comparable across run lengths. Lower is better. Reduction is initial minus final, so positive is improvement.
- Both `muon` and legacy `hybrid` runs prioritize official `torch.optim.Muon` for eligible matrices and use auxiliary AdamW for embeddings, norms, biases, heads, and other ineligible parameters. The local Muon is used only if the official backend is unavailable or incompatible. The `muon` label uses only Muon-managed matrices for the spectral surrogate score; `hybrid` preserves mixed Muon+AdamW scoring.

## Winners by setting

| Setting | Optimizer | Lowest target final | Lowest target AUC | Lowest general final | Lowest general AUC | Largest within-run train reduction |
|---|---|---|---|---|---|---|
| inst_if | AdamW | LayerwiseSoft | LayerwiseSoft | FullTraining | FullTraining | LayerwiseRaw |
| reason_math | AdamW | LayerwiseOptA | LayerwiseSoft | FullTraining | FullTraining | LayerwiseRaw |
| reason_code | AdamW | LayerwiseOptA | LayerwiseOptA | FullTraining | FullTraining | LayerwiseOptA |
| mixed_if | AdamW | LayerwiseSoft | LayerwiseSoft | FullTraining | FullTraining | LayerwiseRaw |
| mixed_math | AdamW | LayerwiseSoft | LayerwiseSoft | FullTraining | FullTraining | LayerwiseRaw |

## Win counts

Counts are across the 5 settings; ties count for every tied method. Raw losses are never averaged across different tasks.

| Optimizer | Method | Target final | Target AUC | General final | General AUC | Available settings |
|---|---|---:|---:|---:|---:|---:|
| AdamW | FullTraining | 0 | 0 | 5 | 5 | 5 |
| AdamW | LayerwiseRaw | 0 | 0 | 0 | 0 | 5 |
| AdamW | LayerwiseSoft | 3 | 4 | 0 | 0 | 5 |
| AdamW | LayerwiseSoftP | 0 | 0 | 0 | 0 | 5 |
| AdamW | LayerwiseOptA | 2 | 1 | 0 | 0 | 5 |

## Detailed loss reductions

### inst_if · AdamW

| Method | Target initial → final (reduction; AUC) | General initial → final (reduction; AUC) | Train early → late (reduction; AUC) | W&B |
|---|---:|---:|---:|---|
| FullTraining | 1.4955 → 1.2598 (0.2357; 1.2760) | 1.6512 → 1.3195 (0.3317; 1.3439) | 1.5101 → 1.2727 (0.2374; 1.3273) | [run](https://wandb.ai/leena12/drpt_opus/runs/jydfac96) |
| LayerwiseRaw | 1.4955 → 1.2595 (0.2360; 1.2770) | 1.6512 → 1.3332 (0.3180; 1.3588) | 1.5293 → 1.2866 (0.2426; 1.3429) | [run](https://wandb.ai/leena12/drpt_opus/runs/12o7szvl) |
| LayerwiseSoft | 1.4955 → 1.2585 (0.2370; 1.2755) | 1.6512 → 1.3308 (0.3203; 1.3559) | 1.5240 → 1.2842 (0.2398; 1.3396) | [run](https://wandb.ai/leena12/drpt_opus/runs/id0jfarx) |
| LayerwiseSoftP | 1.4955 → 1.2869 (0.2087; 1.3089) | 1.6512 → 1.3913 (0.2599; 1.4187) | 1.5782 → 1.3498 (0.2284; 1.4077) | [run](https://wandb.ai/leena12/drpt_opus/runs/0u2s33d2) |
| LayerwiseOptA | 1.4955 → 1.2591 (0.2364; 1.2767) | 1.6512 → 1.3331 (0.3181; 1.3585) | 1.5282 → 1.2867 (0.2416; 1.3428) | [run](https://wandb.ai/leena12/drpt_opus/runs/jt674n37) |

### reason_math · AdamW

| Method | Target initial → final (reduction; AUC) | General initial → final (reduction; AUC) | Train early → late (reduction; AUC) | W&B |
|---|---:|---:|---:|---|
| FullTraining | 0.8615 → 0.6220 (0.2395; 0.6351) | 0.7365 → 0.5043 (0.2322; 0.5204) | 0.6009 → 0.4895 (0.1114; 0.5122) | [run](https://wandb.ai/leena12/drpt_opus/runs/50scw3w1) |
| LayerwiseRaw | 0.8615 → 0.6007 (0.2608; 0.6159) | 0.7365 → 0.5147 (0.2218; 0.5314) | 0.6134 → 0.4994 (0.1141; 0.5230) | [run](https://wandb.ai/leena12/drpt_opus/runs/9shlvmlr) |
| LayerwiseSoft | 0.8615 → 0.5995 (0.2620; 0.6138) | 0.7365 → 0.5127 (0.2238; 0.5291) | 0.6103 → 0.4973 (0.1129; 0.5206) | [run](https://wandb.ai/leena12/drpt_opus/runs/nk2okhhw) |
| LayerwiseSoftP | 0.8614 → 0.6002 (0.2612; 0.6192) | 0.7367 → 0.5570 (0.1798; 0.5756) | 0.6483 → 0.5404 (0.1079; 0.5650) | [run](https://wandb.ai/leena12/drpt_opus/runs/4wlgi92f) |
| LayerwiseOptA | 0.8614 → 0.5993 (0.2621; 0.6149) | 0.7367 → 0.5148 (0.2219; 0.5316) | 0.6136 → 0.4995 (0.1140; 0.5232) | [run](https://wandb.ai/leena12/drpt_opus/runs/vrhfl4gm) |

### reason_code · AdamW

| Method | Target initial → final (reduction; AUC) | General initial → final (reduction; AUC) | Train early → late (reduction; AUC) | W&B |
|---|---:|---:|---:|---|
| FullTraining | 1.8661 → 0.9288 (0.9373; 0.9686) | 0.7367 → 0.5043 (0.2324; 0.5204) | 0.6010 → 0.4895 (0.1115; 0.5122) | [run](https://wandb.ai/leena12/drpt_opus/runs/mlbk63ii) |
| LayerwiseRaw | 1.8661 → 0.6229 (1.2432; 0.7184) | 0.7367 → 0.5152 (0.2215; 0.5321) | 0.6148 → 0.4999 (0.1149; 0.5238) | [run](https://wandb.ai/leena12/drpt_opus/runs/g495jtvk) |
| LayerwiseSoft | 1.8661 → 0.6084 (1.2577; 0.7057) | 0.7367 → 0.5127 (0.2241; 0.5295) | 0.6130 → 0.4975 (0.1155; 0.5212) | [run](https://wandb.ai/leena12/drpt_opus/runs/acyqs5wl) |
| LayerwiseSoftP | 1.8661 → 0.6353 (1.2308; 0.7384) | 0.7367 → 0.5627 (0.1740; 0.5818) | 0.6571 → 0.5454 (0.1117; 0.5706) | [run](https://wandb.ai/leena12/drpt_opus/runs/qtcu1mes) |
| LayerwiseOptA | 1.8661 → 0.6042 (1.2619; 0.7051) | 0.7367 → 0.5151 (0.2217; 0.5324) | 0.6158 → 0.4998 (0.1160; 0.5241) | [run](https://wandb.ai/leena12/drpt_opus/runs/aqnlce7k) |

### mixed_if · AdamW

| Method | Target initial → final (reduction; AUC) | General initial → final (reduction; AUC) | Train early → late (reduction; AUC) | W&B |
|---|---:|---:|---:|---|
| FullTraining | 1.4958 → 1.2667 (0.2292; 1.2821) | 1.0744 → 0.8167 (0.2576; 0.8354) | 0.9663 → 0.8345 (0.1318; 0.8367) | [run](https://wandb.ai/leena12/drpt_opus/runs/gt40928u) |
| LayerwiseRaw | 1.4958 → 1.2645 (0.2313; 1.2809) | 1.0744 → 0.8272 (0.2472; 0.8469) | 0.9822 → 0.8448 (0.1374; 0.8487) | [run](https://wandb.ai/leena12/drpt_opus/runs/otmfgkbf) |
| LayerwiseSoft | 1.4958 → 1.2643 (0.2315; 1.2802) | 1.0744 → 0.8249 (0.2495; 0.8446) | 0.9783 → 0.8425 (0.1357; 0.8460) | [run](https://wandb.ai/leena12/drpt_opus/runs/gyn9thky) |
| LayerwiseSoftP | 1.4958 → 1.2899 (0.2059; 1.3114) | 1.0744 → 0.8771 (0.1973; 0.8978) | 1.0222 → 0.8947 (0.1275; 0.9013) | [run](https://wandb.ai/leena12/drpt_opus/runs/3gh0k83q) |
| LayerwiseOptA | 1.4958 → 1.2646 (0.2312; 1.2813) | 1.0744 → 0.8271 (0.2473; 0.8469) | 0.9818 → 0.8449 (0.1369; 0.8487) | [run](https://wandb.ai/leena12/drpt_opus/runs/qxybf8c3) |

### mixed_math · AdamW

| Method | Target initial → final (reduction; AUC) | General initial → final (reduction; AUC) | Train early → late (reduction; AUC) | W&B |
|---|---:|---:|---:|---|
| FullTraining | 0.8614 → 0.6210 (0.2405; 0.6334) | 1.0744 → 0.8168 (0.2575; 0.8353) | 0.9663 → 0.8345 (0.1317; 0.8367) | [run](https://wandb.ai/leena12/drpt_opus/runs/40o2wtsw) |
| LayerwiseRaw | 0.8614 → 0.5977 (0.2637; 0.6139) | 1.0744 → 0.8301 (0.2442; 0.8494) | 0.9807 → 0.8484 (0.1322; 0.8513) | [run](https://wandb.ai/leena12/drpt_opus/runs/4kbn64dt) |
| LayerwiseSoft | 0.8614 → 0.5960 (0.2654; 0.6109) | 1.0744 → 0.8270 (0.2474; 0.8455) | 0.9747 → 0.8450 (0.1297; 0.8472) | [run](https://wandb.ai/leena12/drpt_opus/runs/tll2ips1) |
| LayerwiseSoftP | 0.8614 → 0.5978 (0.2636; 0.6190) | 1.0744 → 0.8748 (0.1996; 0.8940) | 1.0169 → 0.8926 (0.1243; 0.8968) | [run](https://wandb.ai/leena12/drpt_opus/runs/bga53y7l) |
| LayerwiseOptA | 0.8614 → 0.5964 (0.2651; 0.6127) | 1.0744 → 0.8300 (0.2443; 0.8494) | 0.9800 → 0.8484 (0.1315; 0.8515) | [run](https://wandb.ai/leena12/drpt_opus/runs/pvcgbwox) |

## Missing or incomplete requested runs

None.

## W&B panel recipe

In project `leena12/drpt_opus`, filter by group `<campaign>-<setting>-<optimizer>-s42`, where optimizer is `adamw`, `muon`, or legacy `hybrid`, then filter run names to the methods in this report. Use `train/global_step` as the x-axis and add panels for `train/val_loss` (target), `eval/loss` (general held-out), and `train/loss`. Keep evaluation curves unsmoothed; smoothing is useful only for `train/loss`. Exact URLs and groups are in `wandb_runs.csv`.
