# AdamW/Muon loss-curve comparison

Successful requested runs: **20 / 20**. Missing or incomplete: **0**.

## Reading the metrics

- **Target validation** is `val_loss` on the small target/proxy set used by the selection method. Lower final loss and lower normalized AUC are better, but this set is not independent of selection.
- **General held-out** is `eval_loss` on the larger evaluation set. It is the main loss-based generalization cross-check.
- **Train** compares the mean of the first and last 10% of logged steps; plots use a centered rolling mean. For merged-batch selection runs this is the returned train+target merged loss, while FullTraining returns train-only loss, so cross-method absolute train-loss ranks are not apples-to-apples.
- AUC integrates loss over normalized progress from 0 to 1, making it comparable across run lengths. Lower is better. Reduction is initial minus final, so positive is improvement.
- Both `muon` and legacy `hybrid` runs use the `HybridMuonAdamW` runtime: Muon updates eligible matrix parameters and AdamW updates embeddings, norms, biases, and other ineligible parameters. The `muon` label uses only Muon-managed matrices for the spectral surrogate score; `hybrid` preserves mixed Muon+AdamW scoring.

## Winners by setting

| Setting | Optimizer | Lowest target final | Lowest target AUC | Lowest general final | Lowest general AUC | Largest within-run train reduction |
|---|---|---|---|---|---|---|
| alpaca_samsum | AdamW | LayerwiseSoftP | LayerwiseSoftP | LayerwiseSoftP | LayerwiseSoftP | LayerwiseSoftP |
| less_squad | AdamW | LayerwiseSoftP | LayerwiseSoftP | LayerwiseSoftP | LayerwiseSoftP | FullTraining |
| less_tydiqa | AdamW | LayerwiseSoftP | LayerwiseSoftP | LayerwiseSoftP | LayerwiseSoftP | FullTraining |
| triviaqa_nq | AdamW | LayerwiseSoftP | LayerwiseSoftP | LayerwiseRaw | LayerwiseRaw | LayerwiseSoftP |

## Win counts

Counts are across the four settings; ties count for every tied method. Raw losses are never averaged across different tasks.

| Optimizer | Method | Target final | Target AUC | General final | General AUC | Available settings |
|---|---|---:|---:|---:|---:|---:|
| AdamW | FullTraining | 0 | 0 | 0 | 0 | 4 |
| AdamW | LayerwiseRaw | 0 | 0 | 1 | 1 | 4 |
| AdamW | LayerwiseSoft | 0 | 0 | 0 | 0 | 4 |
| AdamW | LayerwiseSoftP | 4 | 4 | 3 | 3 | 4 |
| AdamW | LayerwiseOptA | 0 | 0 | 0 | 0 | 4 |

## Detailed loss reductions

### alpaca_samsum · AdamW

| Method | Target initial → final (reduction; AUC) | General initial → final (reduction; AUC) | Train early → late (reduction; AUC) | W&B |
|---|---:|---:|---:|---|
| FullTraining | 2.2578 → 1.6596 (0.5982; 1.6491) | 2.0594 → 1.7511 (0.3082; 1.7514) | 1.4732 → 1.4074 (0.0658; 1.3883) | [run](https://wandb.ai/leena12/drpt_opus/runs/u2mjzqt3) |
| LayerwiseRaw | 2.2578 → 1.1885 (1.0693; 1.2566) | 2.0594 → 1.5631 (0.4963; 1.5884) | 1.4895 → 1.4153 (0.0742; 1.3980) | [run](https://wandb.ai/leena12/drpt_opus/runs/49pugpyq) |
| LayerwiseSoft | 2.2578 → 1.1876 (1.0702; 1.2550) | 2.0594 → 1.5623 (0.4971; 1.5875) | 1.4856 → 1.4119 (0.0737; 1.3946) | [run](https://wandb.ai/leena12/drpt_opus/runs/d8gvn7ej) |
| LayerwiseSoftP | 2.2578 → 1.1131 (1.1447; 1.1864) | 2.0594 → 1.5435 (0.5159; 1.5704) | 1.5387 → 1.4575 (0.0812; 1.4391) | [run](https://wandb.ai/leena12/drpt_opus/runs/84z2xj6u) |
| LayerwiseOptA | 2.2578 → 1.1964 (1.0614; 1.2612) | 2.0594 → 1.5662 (0.4932; 1.5905) | 1.4899 → 1.4155 (0.0744; 1.3981) | [run](https://wandb.ai/leena12/drpt_opus/runs/h0nuheeu) |

### less_squad · AdamW

| Method | Target initial → final (reduction; AUC) | General initial → final (reduction; AUC) | Train early → late (reduction; AUC) | W&B |
|---|---:|---:|---:|---|
| FullTraining | 4.6236 → 4.3980 (0.2256; 4.3872) | 4.8678 → 4.5505 (0.3173; 4.5490) | 1.7553 → 1.5158 (0.2395; 1.5639) | [run](https://wandb.ai/leena12/drpt_opus/runs/ye21aqlx) |
| LayerwiseRaw | 4.6236 → 2.9252 (1.6984; 3.1309) | 4.8678 → 3.3101 (1.5577; 3.4608) | 1.8111 → 1.5781 (0.2330; 1.6253) | [run](https://wandb.ai/leena12/drpt_opus/runs/j72so9z8) |
| LayerwiseSoft | 4.6236 → 3.0093 (1.6144; 3.1470) | 4.8678 → 3.3668 (1.5011; 3.4745) | 1.8056 → 1.5690 (0.2367; 1.6167) | [run](https://wandb.ai/leena12/drpt_opus/runs/5olplzso) |
| LayerwiseSoftP | 4.6236 → 2.5514 (2.0722; 2.6879) | 4.8678 → 3.0673 (1.8006; 3.1404) | 1.8778 → 1.6531 (0.2247; 1.6965) | [run](https://wandb.ai/leena12/drpt_opus/runs/stresysc) |
| LayerwiseOptA | 4.6236 → 2.9433 (1.6803; 3.1245) | 4.8678 → 3.3164 (1.5515; 3.4560) | 1.8120 → 1.5796 (0.2324; 1.6267) | [run](https://wandb.ai/leena12/drpt_opus/runs/i196nk33) |

### less_tydiqa · AdamW

| Method | Target initial → final (reduction; AUC) | General initial → final (reduction; AUC) | Train early → late (reduction; AUC) | W&B |
|---|---:|---:|---:|---|
| FullTraining | 1.3386 → 1.0848 (0.2538; 1.0670) | 1.9630 → 1.6114 (0.3516; 1.5823) | 1.7554 → 1.5160 (0.2393; 1.5640) | [run](https://wandb.ai/leena12/drpt_opus/runs/dvmzo7ml) |
| LayerwiseRaw | 1.3386 → 0.4742 (0.8644; 0.5543) | 1.9630 → 0.7294 (1.2336; 0.8305) | 1.7603 → 1.5311 (0.2292; 1.5773) | [run](https://wandb.ai/leena12/drpt_opus/runs/d92m5uy3) |
| LayerwiseSoft | 1.3386 → 0.4851 (0.8535; 0.5593) | 1.9630 → 0.7553 (1.2077; 0.8362) | 1.7558 → 1.5201 (0.2357; 1.5671) | [run](https://wandb.ai/leena12/drpt_opus/runs/g0wfhj0w) |
| LayerwiseSoftP | 1.3386 → 0.3574 (0.9812; 0.4286) | 1.9630 → 0.5913 (1.3717; 0.6546) | 1.8302 → 1.6102 (0.2200; 1.6522) | [run](https://wandb.ai/leena12/drpt_opus/runs/4ign0q18) |
| LayerwiseOptA | 1.3386 → 0.5236 (0.8150; 0.5823) | 1.9630 → 0.7821 (1.1809; 0.8575) | 1.7603 → 1.5348 (0.2255; 1.5791) | [run](https://wandb.ai/leena12/drpt_opus/runs/t72b7s29) |

### triviaqa_nq · AdamW

| Method | Target initial → final (reduction; AUC) | General initial → final (reduction; AUC) | Train early → late (reduction; AUC) | W&B |
|---|---:|---:|---:|---|
| FullTraining | 4.1855 → 1.9659 (2.2196; 2.0149) | 5.1573 → 2.6072 (2.5501; 2.6519) | 2.0048 → 1.3230 (0.6818; 1.4429) | [run](https://wandb.ai/leena12/drpt_opus/runs/h553mzpe) |
| LayerwiseRaw | 4.1855 → 1.5983 (2.5872; 1.7063) | 5.1573 → 2.4667 (2.6906; 2.5268) | 2.1038 → 1.4290 (0.6748; 1.5410) | [run](https://wandb.ai/leena12/drpt_opus/runs/pmj1dqq0) |
| LayerwiseSoft | 4.1855 → 1.6093 (2.5762; 1.7207) | 5.1573 → 2.4846 (2.6727; 2.5399) | 2.0903 → 1.4268 (0.6634; 1.5336) | [run](https://wandb.ai/leena12/drpt_opus/runs/kuhphp51) |
| LayerwiseSoftP | 4.1855 → 1.5570 (2.6285; 1.6751) | 5.1573 → 2.4723 (2.6850; 2.5338) | 2.2128 → 1.4770 (0.7358; 1.5929) | [run](https://wandb.ai/leena12/drpt_opus/runs/8zt9s80r) |
| LayerwiseOptA | 4.1855 → 1.5891 (2.5964; 1.7035) | 5.1573 → 2.4714 (2.6859; 2.5283) | 2.1056 → 1.4312 (0.6744; 1.5393) | [run](https://wandb.ai/leena12/drpt_opus/runs/ifa3vrvb) |

## Missing or incomplete requested runs

None.

## W&B panel recipe

In project `leena12/drpt_opus`, filter by group `<campaign>-<setting>-<optimizer>-s42`, where optimizer is `adamw`, `muon`, or legacy `hybrid`, then filter run names to the methods in this report. Use `train/global_step` as the x-axis and add panels for `train/val_loss` (target), `eval/loss` (general held-out), and `train/loss`. Keep evaluation curves unsmoothed; smoothing is useful only for `train/loss`. Exact URLs and groups are in `wandb_runs.csv`.
