# Campaign analysis

Post-hoc analysis of a finished SFT campaign. Everything here reads artifacts
that completed runs already wrote — except the layer-alignment probe, which is
the one piece that needs a GPU (and still does no training).

Outputs land in `results/<campaign-id>/`.

## Three questions, three entry points

| Script | Question | Needs |
|---|---|---|
| `selection_profile.py` | What weight did each method assign, and what data did it actually consume? | completed runs only |
| `axis_ablation.py` | How does target val loss move along the data / architecture / optimizer-geometry axes? | completed runs only |
| `layer_alignment_probe.py` → `layer_alignment_report.py` | Is a per-layer-group weight `w_l` derivable from target gradients, and does it separate by target? | 1 GPU, ~30 min, no training |

```bash
# CPU, seconds
python -m SFT.eval.analysis.selection_profile --campaign dolci32k-qwen3_1_7b-s42
python -m SFT.eval.analysis.axis_ablation     --campaign dolci32k-qwen3_1_7b-s42

# GPU probe, then its report
sbatch ... SFT/eval/analysis/probe_job.sh
python -m SFT.eval.analysis.layer_alignment_report --split val
```

All three are campaign-parameterised, so they re-run unchanged on
`dolci32k-qwen3_4b-s42` once that campaign completes.

New Dolci runs under the current defaults write `selection_records.json` every
100 steps. It
contains `train_meta_idx`, decoded candidates, and each layer's exact hard
indices or continuous weights, so layer/sample identity comparisons can be
joined across methods without Qwen32k files. Older completed campaigns may not
contain this artifact and need a rerun for exact identity analysis.
`selection_profile.py` covers the
AdamW registry plus the separate Global axis and all four Muon surrogate
variants; aggregate domain/source selection rate and lift come from
`selection_domain_summary.json`.

AdamW and Muon selection reports are written separately under
`selection_profile/adamw/` and `selection_profile/muon/`, so running one family
does not overwrite the other. Each contains `campaign_completeness.*`, which
lists every expected setting/method cell as complete or missing.

## What each script assumes

`campaign_io.py` holds the one modelling choice shared by all of them: the
three-axis reading of each method (`METHOD_AXES`). Everything else is a file
read. Runs are enumerated from `run_status.json`, never by parsing directory
names.

`viz.py` fixes one hue per method, assigned in registry order and never cycled,
so a method keeps its identity across every figure. The categorical subset is
validated on the light surface; the sub-3:1 contrast warning is relieved by
shipping the same numbers as CSV/Markdown beside every figure.

## Reading the results

Three things change how the numbers should be read, and each is repeated in the
output it applies to:

1. **The hard methods keep 8 of 16, not 4.** dolci32k runs `batch_size: 16`,
   `selection_frac: 0.5`. Top-4 was the older `baseline9`/`loss52` setting.
2. **`lift` is already normalised** by each run's own overall selection rate, so
   `LayerwiseSoftP` (mass 1) compares directly to the hard methods (mass 8).
3. **The deltas are small and single-seed.** `axis_ablation` prints a noise
   floor from each curve's own late-window wobble and labels anything below it
   as within noise. That floor is a lower bound, not a confidence interval —
   only replication across seeds would give one.

## The pending cells

`GlobalRaw` and `GlobalOptA` supply the architecture axis (global vs layer-wise
selection). They are not part of the immutable `ADAMW_METHODS` registry, so they
run as a separate array — see `SFT/train/submit_dolci32k_global_axis.sh`. Until
those finish, `axis_ablation` renders them as pending and `selection_profile`
marks them missing in `campaign_completeness.*`; all available cells still
produce their normal tables and figures.
