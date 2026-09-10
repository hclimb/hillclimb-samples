# Is layer-wise selection actually layer-wise?

Recovered from the 25 completed dolci32k AdamW runs. Nothing re-trained.

## Q1. Did different layers choose different samples?

**w** = the fraction of layers that kept a given candidate (1.0 = every layer kept it, 0.0 = none did).

**How to read the last three columns.** `boundary(obs)` is the share of w values sitting at 0 or 1. `boundary(if identical)` is what that share *would* be if every layer behaved the same way -- 100% for the hard top-k methods (a 0/1 vote is always at the boundary), and the run's own logged within-layer `soft/boundary_fraction` for the soft methods. **A large gap between the two means the layers disagreed.**

| setting | method | n | median w | SD of w | boundary(obs) | boundary(if identical) | gap |
|---|---|---:|---:|---:|---:|---:|---:|
| inst_if | LayerwiseRaw | 250 | 0.540 | 0.195 | 0.0% | 100.0% | +100.0pp |
| inst_if | LayerwiseOptA | 250 | 0.543 | 0.239 | 0.4% | 100.0% | +99.6pp |
| inst_if | LayerwiseSoft | 250 | 0.495 | 0.219 | 0.0% | 43.1% | +43.1pp |
| inst_if | LayerwiseSoftP | 250 | 0.022 | 0.138 | 47.2% | 92.9% | +45.7pp |
| reason_math | LayerwiseRaw | 210 | 0.540 | 0.206 | 0.0% | 100.0% | +100.0pp |
| reason_math | LayerwiseOptA | 210 | 0.540 | 0.238 | 1.0% | 100.0% | +99.0pp |
| reason_math | LayerwiseSoft | 210 | 0.521 | 0.211 | 0.5% | 44.3% | +43.9pp |
| reason_math | LayerwiseSoftP | 210 | 0.085 | 0.155 | 17.6% | 89.3% | +71.7pp |
| reason_code | LayerwiseRaw | 210 | 0.513 | 0.218 | 0.0% | 100.0% | +100.0pp |
| reason_code | LayerwiseOptA | 210 | 0.505 | 0.246 | 0.0% | 100.0% | +100.0pp |
| reason_code | LayerwiseSoft | 210 | 0.483 | 0.214 | 0.0% | 48.4% | +48.4pp |
| reason_code | LayerwiseSoftP | 210 | 0.058 | 0.208 | 26.2% | 93.6% | +67.4pp |
| mixed_if | LayerwiseRaw | 542 | 0.525 | 0.201 | 0.0% | 100.0% | +100.0pp |
| mixed_if | LayerwiseOptA | 542 | 0.525 | 0.249 | 0.7% | 100.0% | +99.3pp |
| mixed_if | LayerwiseSoft | 542 | 0.509 | 0.224 | 0.6% | 39.6% | +39.0pp |
| mixed_if | LayerwiseSoftP | 542 | 0.031 | 0.143 | 42.4% | 92.3% | +49.9pp |
| mixed_math | LayerwiseRaw | 542 | 0.475 | 0.206 | 0.0% | 100.0% | +100.0pp |
| mixed_math | LayerwiseOptA | 542 | 0.480 | 0.246 | 0.9% | 100.0% | +99.1pp |
| mixed_math | LayerwiseSoft | 542 | 0.496 | 0.234 | 0.7% | 46.2% | +45.5pp |
| mixed_math | LayerwiseSoftP | 542 | 0.022 | 0.110 | 48.3% | 92.2% | +43.8pp |

For reference, if layers picked **independently at random** the SD of w would be about 0.036 (Binomial(U≈198, 0.5)/U) -- i.e. essentially every w crowded onto 0.5.

## Q2a. Do the architecture groups differ?

Median selected-vs-unselected **score margin** over all 2000 steps. Bigger = that part of the network separates good from bad candidates more sharply. Near zero = it barely distinguishes them.

Only the hard top-k methods log this; the soft methods log solver diagnostics instead.

| setting | method | embedding | attention | mlp | lm_head | lm_head ÷ attention |
|---|---|---:|---:|---:|---:|---:|
| inst_if | LayerwiseRaw | 5.37e-03 | 1.90e-03 | 5.74e-03 | 9.40e-02 | 49x |
| inst_if | LayerwiseOptA | 1.50e-05 | 3.24e-05 | 5.05e-05 | 1.78e-04 | 5x |
| reason_math | LayerwiseRaw | 7.41e-03 | 1.14e-03 | 3.77e-03 | 6.92e-02 | 61x |
| reason_math | LayerwiseOptA | 1.75e-05 | 2.63e-05 | 5.01e-05 | 1.19e-04 | 5x |
| reason_code | LayerwiseRaw | 1.66e-02 | 2.40e-03 | 7.33e-03 | 8.84e-02 | 37x |
| reason_code | LayerwiseOptA | 3.27e-05 | 4.83e-05 | 8.09e-05 | 1.74e-04 | 4x |
| mixed_if | LayerwiseRaw | 4.13e-03 | 1.42e-03 | 4.50e-03 | 7.06e-02 | 50x |
| mixed_if | LayerwiseOptA | 1.16e-05 | 2.46e-05 | 3.90e-05 | 1.29e-04 | 5x |
| mixed_math | LayerwiseRaw | 8.31e-03 | 1.34e-03 | 4.56e-03 | 7.90e-02 | 59x |
| mixed_math | LayerwiseOptA | 1.58e-05 | 2.58e-05 | 4.40e-05 | 1.56e-04 | 6x |

## Q2b. Does each task pull a different data mix?

`lift` of the **target domain** = how much more often that domain is kept than its share of the candidate pool. 1.00 = no preference at all; >1 = the method steers toward the target.

| setting | target domain | LayerwiseRaw | LayerwiseOptA | LayerwiseSoft | LayerwiseSoftP |
|---|---|---:|---:|---:|---:|
| inst_if | `precise_if` | 1.15 | 1.16 | 1.13 | 1.50 |
| reason_math | `math` | 0.89 | 0.88 | 0.93 | 0.61 |
| reason_code | `mbpp` | not in pool | not in pool | not in pool | not in pool |
| mixed_if | `precise_if` | 1.25 | 1.27 | 1.21 | 1.94 |
| mixed_math | `math` | 0.90 | 0.87 | 0.88 | 0.41 |

Where a cell says *not in pool*, that domain label does not exist in that setting's candidate pool at all, so no steering toward it is even possible.

![layer agreement](layerwise_analysis.png)

## What is NOT recoverable from these runs

Per-*individual*-layer sample identity. The campaign ran without `--record_selections`, so `selection_records.json` -- which would store each layer's own selected indices -- exists in none of the 25 runs. Statements like "layer 3 preferred domain X, layer 27 preferred Y" cannot be made; only the 4 optimizer-group buckets in Q2a. Re-running one setting with `--record_selections` would close that gap.

