# Dolci32k Method Configs

The nine registry YAML files here cover all 13 Dolci32k baselines:

| Family | Methods |
|---|---|
| AdamW (5) | `FullTraining`, `LayerwiseRaw`, `LayerwiseSoft`, `LayerwiseSoftP`, `LayerwiseOptA` |
| Muon (8) | `FullTraining`, `LayerwiseRaw`, `LayerwiseSoft`, `LayerwiseSoftP`, `LayerwiseMuonSur`, `LayerwiseMuonPSur`, `LayerwiseMuonSatSur`, `LayerwiseMuonSatPSur` |

The four AdamW/Muon methods in common resolve to the same YAML, so nine files
suffice. These configs inherit the original `configs/less_tydiqa/` method
settings. The two Soft configs intentionally add
`soft_weighting.replay_precision: bf16_fp32` for the Dolci32k profile. The
continuous solver weights, target, normalization, reductions, and outputs stay
FP32 while the large Muon Soft solver/replay factor contractions use BF16
operands. Legacy configs keep the exact `fp32` contraction default.

`GlobalSubset-Full.yaml` and `OptimizerAwareGlobalSubset-Full.yaml` are two
further copies of the same shared method configs (verified semantically
identical across all four original setting dirs). They are **not** part of the
immutable `ADAMW_METHODS`/`MUON_METHODS` registry in
`SFT/data/dolci32k/profile.py`, which stays at 5/8 so the `0-24` and
`0-39` array contracts are unchanged. They exist for the separate
`GlobalRaw`/`GlobalOptA` axis array
(`SFT/train/submit_dolci32k_global_axis.sh`), which supplies the
architecture-axis control (global vs layer-wise selection) that the main
campaign omits.

**There is no `defaults.yaml` here.** Each of the six extended settings owns a
thin directory containing only its own `defaults.yaml`, which points back here:

```yaml
method_config_dir: configs/dolci32k_methods
```

`SFT/train/train.sh` resolves a method YAML by looking in the setting's own
directory first and falling back to `method_config_dir`. That keeps the original
four settings untouched while letting the six new ones share these nine files
instead of duplicating 54 copies.

Because a setting directory holds no method YAMLs of its own, the auto-discovery
categories that glob `$config_dir/*.yaml` (`all`, `full`, `lora`, ...) resolve to
nothing for these settings. Use explicit method names or the `sft10` /
`baseline9-adamw` / `baseline9-muon` categories, which resolve through
`method_config_exists` and therefore honour the fallback directory.
