# Downstream evaluation · dolci32k-qwen3_1_7b-s42 · adamw

Task-native metrics are reported separately; no cross-task mean is computed.

## Status

| Status | Count |
|---|---:|
| already_evaluated | 10 |
| evaluated | 25 |

## Results

| Setting | Task | Method | Primary metric | Value |
|---|---|---|---|---:|
| inst_if | ifbench | FullTraining | prompt_level_loose_acc | 22.000000 |
| inst_if | ifeval | FullTraining | prompt_level_strict_acc | 38.262477 |
| inst_if | ifbench | LayerwiseOptA | prompt_level_loose_acc | 18.000000 |
| inst_if | ifeval | LayerwiseOptA | prompt_level_strict_acc | 33.641405 |
| inst_if | ifbench | LayerwiseRaw | prompt_level_loose_acc | 18.333333 |
| inst_if | ifeval | LayerwiseRaw | prompt_level_strict_acc | 35.120148 |
| inst_if | ifbench | LayerwiseSoft | prompt_level_loose_acc | 19.666667 |
| inst_if | ifeval | LayerwiseSoft | prompt_level_strict_acc | 41.219963 |
| inst_if | ifbench | LayerwiseSoftP | prompt_level_loose_acc | 19.666667 |
| inst_if | ifeval | LayerwiseSoftP | prompt_level_strict_acc | 35.304991 |
| mixed_if | ifbench | FullTraining | prompt_level_loose_acc | 17.666667 |
| mixed_if | ifeval | FullTraining | prompt_level_strict_acc | 38.817006 |
| mixed_if | ifbench | LayerwiseOptA | prompt_level_loose_acc | 19.000000 |
| mixed_if | ifeval | LayerwiseOptA | prompt_level_strict_acc | 31.238447 |
| mixed_if | ifbench | LayerwiseRaw | prompt_level_loose_acc | 16.666667 |
| mixed_if | ifeval | LayerwiseRaw | prompt_level_strict_acc | 31.977819 |
| mixed_if | ifbench | LayerwiseSoft | prompt_level_loose_acc | 19.666667 |
| mixed_if | ifeval | LayerwiseSoft | prompt_level_strict_acc | 36.783734 |
| mixed_if | ifbench | LayerwiseSoftP | prompt_level_loose_acc | 17.333333 |
| mixed_if | ifeval | LayerwiseSoftP | prompt_level_strict_acc | 32.717190 |
| mixed_math | math500 | FullTraining | accuracy | 56.800000 |
| mixed_math | math500 | LayerwiseOptA | accuracy | 32.200000 |
| mixed_math | math500 | LayerwiseRaw | accuracy | 13.400000 |
| mixed_math | math500 | LayerwiseSoft | accuracy | 29.600000 |
| mixed_math | math500 | LayerwiseSoftP | accuracy | 0.000000 |
| reason_code | mbpp_plus | FullTraining | base_plus_extra_pass_at_1 | 53.968254 |
| reason_code | mbpp_plus | LayerwiseOptA | base_plus_extra_pass_at_1 | 35.714286 |
| reason_code | mbpp_plus | LayerwiseRaw | base_plus_extra_pass_at_1 | 37.037037 |
| reason_code | mbpp_plus | LayerwiseSoft | base_plus_extra_pass_at_1 | 53.968254 |
| reason_code | mbpp_plus | LayerwiseSoftP | base_plus_extra_pass_at_1 | 33.597884 |
| reason_math | math500 | FullTraining | accuracy | 56.600000 |
| reason_math | math500 | LayerwiseOptA | accuracy | 51.600000 |
| reason_math | math500 | LayerwiseRaw | accuracy | 48.600000 |
| reason_math | math500 | LayerwiseSoft | accuracy | 53.000000 |
| reason_math | math500 | LayerwiseSoftP | accuracy | 39.400000 |
