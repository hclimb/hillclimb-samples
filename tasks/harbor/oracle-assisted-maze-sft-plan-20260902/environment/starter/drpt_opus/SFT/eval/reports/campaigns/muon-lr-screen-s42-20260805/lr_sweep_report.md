# Muon LR screening: muon-lr-screen-s42-20260805

Completed cells: **20 / 20 discovered**.

This is a short max-step screen. Lower held-out AUC/final loss is better. The winning LR must be confirmed with a full-length run because the short Trainer schedule decays over the screening horizon.

| Muon LR | Completed tasks | Mean per-task AUC rank | Mean relative AUC |
|---:|---:|---:|---:|
| 3.0e-04 | 4 | 1.500 | 0.80449 |
| 1.0e-03 | 4 | 2.000 | 0.82825 |
| 1.0e-04 | 4 | 2.500 | 0.86121 |
| 3.0e-05 | 4 | 4.000 | 0.96376 |
| 1.0e-05 | 4 | 5.000 | 0.97881 |

## Per-task results

### alpaca_samsum

| Muon LR | Status | Final eval loss | Normalized AUC | AUC rank | Last step |
|---:|---|---:|---:|---:|---:|
| 1.0e-05 | completed | 2.029565 | 2.036434 | 5 | 200 |
| 3.0e-05 | completed | 2.008455 | 2.021488 | 4 | 200 |
| 1.0e-04 | completed | 1.884567 | 1.922995 | 3 | 200 |
| 3.0e-04 | completed | 1.786402 | 1.820296 | 1 | 200 |
| 1.0e-03 | completed | 1.912382 | 1.914570 | 2 | 200 |

### less_squad

| Muon LR | Status | Final eval loss | Normalized AUC | AUC rank | Last step |
|---:|---|---:|---:|---:|---:|
| 1.0e-05 | completed | 4.788740 | 4.806765 | 5 | 200 |
| 3.0e-05 | completed | 4.735264 | 4.766980 | 4 | 200 |
| 1.0e-04 | completed | 4.412407 | 4.511510 | 1 | 200 |
| 3.0e-04 | completed | 4.701158 | 4.608395 | 3 | 200 |
| 1.0e-03 | completed | 4.625268 | 4.602809 | 2 | 200 |

### less_tydiqa

| Muon LR | Status | Final eval loss | Normalized AUC | AUC rank | Last step |
|---:|---|---:|---:|---:|---:|
| 1.0e-05 | completed | 1.923053 | 1.934262 | 5 | 200 |
| 3.0e-05 | completed | 1.893110 | 1.910994 | 4 | 200 |
| 1.0e-04 | completed | 1.666253 | 1.735968 | 3 | 200 |
| 3.0e-04 | completed | 1.682021 | 1.658359 | 1 | 200 |
| 1.0e-03 | completed | 1.714223 | 1.684707 | 2 | 200 |

### triviaqa_nq

| Muon LR | Status | Final eval loss | Normalized AUC | AUC rank | Last step |
|---:|---|---:|---:|---:|---:|
| 1.0e-05 | completed | 4.835138 | 4.908988 | 5 | 200 |
| 3.0e-05 | completed | 4.601125 | 4.739555 | 4 | 200 |
| 1.0e-04 | completed | 3.117699 | 3.602739 | 3 | 200 |
| 3.0e-04 | completed | 2.591814 | 2.791979 | 1 | 200 |
| 1.0e-03 | completed | 2.976391 | 2.982624 | 2 | 200 |
