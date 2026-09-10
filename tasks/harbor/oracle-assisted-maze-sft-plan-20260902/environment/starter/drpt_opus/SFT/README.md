# SFT Experiments

Training and evaluation code for Supervised Fine-Tuning experiments.
See `PROGRESS.md` for the live run inventory and status.

## Active pipelines

There are two SFT launch paths with different purposes:

- `SFT/train/submit_general_loss_comparison.sh` is the canonical optimizer
  comparison. Its default `baseline9` profile runs exactly 5 AdamW and 8 Muon
  baselines on each of the four settings at seed 42: 20 AdamW + 32 Muon = 52
  training runs. The canonical, model-independent `dolci32k` profile uses the
  same method families on five settings: 25 AdamW + 40 Muon = 65 training runs
  for each selected model profile.
- `SFT/train/submit_all.sh` is the legacy five-seed Full/LoRA/MeSO study. It
  also trains target-only controls and submits downstream evaluation jobs. It
  is not the 5 AdamW + 8 Muon comparison.

`sft_drpt_opus.sh` is a single-setting convenience wrapper for older method
bundles. Use the focused launcher above when reproducing the optimizer
comparison.

## Focused AdamW/Muon comparison

The default profile uses full-model SFT for every method:

| Family | Baselines per setting |
|---|---|
| AdamW (5) | `FullTraining`, `LayerwiseRaw`, `LayerwiseSoft`, `LayerwiseSoftP`, `LayerwiseOptA` |
| Muon (8) | `FullTraining`, `LayerwiseRaw`, `LayerwiseSoft`, `LayerwiseSoftP`, `LayerwiseMuonSur`, `LayerwiseMuonPSur`, `LayerwiseMuonSatSur`, `LayerwiseMuonSatPSur` |

The Muon family uses `optimizer_type=muon`. Eligible two-dimensional hidden
weights are updated by official `torch.optim.Muon` whenever it is available;
embeddings, norms, biases, output heads, and other ineligible parameters use an
auxiliary official AdamW optimizer. The single Trainer-facing optimizer object
is only a facade coordinating these official optimizers; it does not reimplement
the normal Muon update. The existing local Muon implementation is used only if
the installed official Muon is unavailable or incompatible. For the four Muon
surrogate methods, selection scores use Muon-managed matrices only.

Legacy mixed-score selection ablations are named explicitly, for example
`LayerwiseHybridMuonSur` and `LayerwiseHybridMuonMatrixSur`. Their `Hybrid`
label describes which parameter groups contribute to ranking, not a custom
replacement for `torch.optim.Muon`; they are not part of the baseline9 Muon
family.

`LayerwiseSoft` uses a capped simplex with total weight 4 for a full batch of
8. `LayerwiseSoftP` uses a probability simplex with total weight 1. The four
Muon surrogates form the uniform/singular-value-weighted x
unsaturated/saturated matrix-score variants.

```bash
# Inspect the exact 52-run matrix without submitting anything.
bash SFT/train/submit_general_loss_comparison.sh \
  --campaign-id <unique-id> --dry-run

# Submit independent AdamW 0-19 and Muon 0-31 arrays (maximum 5 concurrent).
bash SFT/train/submit_general_loss_comparison.sh \
  --campaign-id <unique-id>

# A single family can be launched or retried independently.
bash SFT/train/submit_general_loss_comparison.sh \
  --campaign-id <unique-id> --family adamw
```

The launcher submits one array per optimizer family. Each array task maps its
index to a setting/method pair and calls `SFT/train/train.sh`, which resolves
the short method label to a YAML config, applies the optimizer override, and
invokes `SFT/train/train.py`. A family-specific loss-report job runs with an
`afterany` dependency, so the AdamW and Muon reports do not wait for each
other. Downstream generation metrics are a separate optional stage via
`SFT/eval/submit_campaign_downstream.sh`.

The launcher also runs an idempotent data preparation step before submission.
It downloads/converts only missing source groups, then all array workers perform
a cheap 14-file preflight:

```bash
python SFT/data/prepare_baseline9.py
python SFT/data/prepare_baseline9.py --check-only
```

Set `DRPT_DATA_DIR` to keep prepared JSONL files outside the checkout.

## Unified Dolci-Instruct 32K benchmark (`--profile dolci32k`)

`dolci32k` is the canonical five-setting SFT profile. Its only general-pool
source is the immutable
`allenai/Dolci-Instruct-SFT@bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221`
release. It does not load a separate OpenThoughts corpus or Dolci-Think. The
source named `Dolci Instruct OpenThoughts3+ Science` below is already part of
the released Dolci-Instruct mixture.

Raw pool membership and candidate order are model-independent. Every model,
optimizer, selection method, and comparable run loads the same stable example
IDs; only the tokenizer/chat-template-derived cache differs. The supported
pinned base-model overlays are:

| Model profile | Base checkpoint | Tokenizer |
|---|---|---|
| `olmo3_7b` | `allenai/Olmo-3-1025-7B@a81bae42db3975be1671e27b9c9a56da1a9f980f` | `allenai/olmo-3-tokenizer-instruct-release@134a8a9166b45a96f21cf1e66b928fe6e6c0b9b1` |
| `qwen3_1_7b` | `Qwen/Qwen3-1.7B-Base@ea980cb0a6c2ae4b936e82123acc929f1cec04c1` | same pinned repository/revision |
| `qwen3_4b` | `Qwen/Qwen3-4B-Base@906bfd4b4dc7f14ee4320094d8b41684abff8539` | same pinned repository/revision |
| `qwen3_8b` | `Qwen/Qwen3-8B-Base@49e3418fbbbca6ecbdf9608b4d22e5a407081db4` | same pinned repository/revision |

### Verified upstream metadata

The builder independently reproduces the complete 2,152,112-row inventory
before sampling. The pinned release contains exactly 11 domains and 22 sources,
with one domain per `source_dataset`:

| Domain | Rows | `source_dataset` rows |
|---|---:|---|
| Chat | 309,538 | Wildchat 302,406; OpenAssistant 7,132 |
| Coding | 328,614 | Dolci Instruct Python Algorithms 186,345; Evol CodeAlpaca 107,270; Tulu 3 Persona Python 34,999 |
| Hardcoded Data | 69 | Hardcoded Data 69 |
| Math | 269,937 | Tulu 3 Persona MATH 149,958; OpenMathInstruct 2 50,000; Tulu 3 Persona GSM 49,980; Tulu 3 Persona Algebra 19,999 |
| Multilingual | 99,987 | Aya 99,987 |
| Other | 254,863 | Logic Puzzles 159,882; FLAN 89,981; TableGPT 5,000 |
| Precise IF | 136,833 | Dolci Instruct Precise IF 136,833 |
| Reasoning | 310,572 | Verifiable Reasoning 310,572 |
| Safety | 110,295 | WildJailbreak 49,965; WildGuardMix 49,373; CoCoNot 10,957 |
| Science | 103,825 | Dolci Instruct OpenThoughts3+ Science 99,268; SciRiff 4,557 |
| Tool Use | 227,579 | Dolci Instruct Tool Use 227,579 |

Any unseen domain, source, or domain-source pairing fails closed. The complete
inventory, domain/source/cross counts and percentages, representative examples,
allocation quotas, lineage, hashes, and overlap matrices are saved with the
build in `pool_statistics.json` and `manifest.json`.

### Pool construction

Targets and final benchmarks are reserved first, Reasoning is built second,
and its stable IDs and normalized prompts are reserved before Instruction is
sampled. Thus Instruction and Reasoning are disjoint parents even though some
eligible source families appear in both definitions.

Instruction-32K excludes the `Math`, `Coding`, `Science`, and `Tool Use` domains and also
rejects structurally detected tool/function-calling rows. Its eligible sources
are Wildchat, OpenAssistant, FLAN, Logic Puzzles, TableGPT, Aya, Dolci Instruct
Precise IF, Verifiable Reasoning, WildJailbreak, WildGuardMix, CoCoNot, and
Hardcoded Data. Both Science sources remain in Reasoning-32K with their existing
quotas, but no Science-domain row is eligible for Instruction-32K. Precise-IF is
not removed wholesale: only the exact reserved target examples are excluded.
A deterministic capped-equal water-filling allocator samples the 32,512 parent
rows without replacement, redistributes source shortages, and then assigns 512
validation rows by largest-remainder source stratification. The remaining
32,000 rows are training data.

Reasoning-32K uses exact, source-balanced quotas:

| Category | Sources | Train/source | Val/source | Category train/val |
|---|---|---:|---:|---:|
| Math | Tulu 3 Persona MATH; OpenMathInstruct 2; Tulu 3 Persona GSM; Tulu 3 Persona Algebra | 3,000 | 48 | 12,000 / 192 |
| Coding | Dolci Instruct Python Algorithms; Evol CodeAlpaca; Tulu 3 Persona Python | 4,000 | 64 | 12,000 / 192 |
| Science/Logic/Verifiable | Verifiable Reasoning; Logic Puzzles; Dolci Instruct OpenThoughts3+ Science; SciRiff | 2,000 | 32 | 8,000 / 128 |

Mixed-32K is never resampled from upstream Dolci. It is a deterministic,
stratified nested subset of the finalized parents: 16,000 Instruction plus
16,000 Reasoning for training, and 256 Instruction plus 256 Reasoning for
validation. Instruction is stratified by source; Reasoning is stratified by
category and source. Parent membership and lineage are asserted, so sources
such as Precise-IF retain their parent proportions as closely as integer quotas
allow.

The five public settings are:

| Config | General train/val | Target grad/val | Final evaluation |
|---|---|---|---|
| `configs/dolci32k/inst_if` | Instruction-32K (32,000 / 512) | Precise-IF (64 / 128) | IFEval + IFBench |
| `configs/dolci32k/reason_math` | Reasoning-32K (32,000 / 512) | MATH train (64 / 128) | MATH500 |
| `configs/dolci32k/reason_code` | Reasoning-32K (32,000 / 512) | MBPP train (64 / 128) | MBPP+ |
| `configs/dolci32k/mixed_if` | Mixed-32K (32,000 / 512) | same Precise-IF artifacts as `inst_if` | IFEval + IFBench |
| `configs/dolci32k/mixed_math` | Mixed-32K (32,000 / 512) | same MATH artifacts as `reason_math` | MATH500 |

The 64-row target-gradient split remains selection-only, and the separate
128-row target-validation split retains the existing monitoring semantics.
Stable IDs and normalized-prompt exact matches are disjoint across each target
and relevant general train/validation data; the existing word-8-gram leakage
check is retained as an additional safeguard. MBPP target selection also
excludes MBPP+ task IDs.

### Immutable artifacts and tokenizer diagnostics

Builds are content-addressed and promoted atomically under
`$DRPT_DATA_DIR/dolci32k_artifacts/builds/<build-id>/`; `CURRENT` advances only
after all count, uniqueness, disjointness, lineage, and hash assertions pass.
The manifest stores stable IDs, upstream indices and revisions, roles, domains,
sources, quotas, ordered candidate hashes, and raw-file hashes.

```bash
# Print all domain, source, and domain x source counts and verify the pin.
python SFT/data/prepare_dolci32k.py --inspect-metadata

# Build/audit immutable raw manifests, then profile all four model aliases.
python SFT/data/prepare_dolci32k.py --build --profile-tokenizers all

# Cheap hash/count audit of CURRENT (no raw resampling).
python SFT/data/prepare_dolci32k.py --audit-only

# Recompute overlap/lineage assertions from artifact rows.
python SFT/data/prepare_dolci32k.py --reaudit

# Add or refresh tokenizer diagnostics for an already-built CURRENT artifact.
python SFT/data/prepare_dolci32k.py \
  --audit-only --profile-tokenizers all
```

Tokenizer profiling never changes raw membership. For every general and target
split, each alias reports untruncated length p50/p90/p99/max, fraction over
4,096, and the assistant-token/truncation breakdown by `source_dataset`. The
canonical `right_then_assistant_tail` policy separately reports (1) rows that
would have zero assistant supervision under ordinary right truncation, (2) rows
where the deterministic assistant-tail fallback was applied, and (3) rows that
still have zero supervision after that fallback. A warning is emitted when more
than 5% of any split is truncated. Per-example rows are cached in Parquet by raw
build ID, artifact hashes, tokenizer/template fingerprint, and maximum length;
Qwen3-4B and Qwen3-8B can share one physical cache when their tokenizer behavior
matches, while both aliases retain separate report entries. The derived report is stored below
`dolci32k_artifacts/tokenization_reports/<build-id>/`. Training preflight checks
the report and Parquet hashes and fails with affected IDs/source counts if any
required row still has zero assistant supervision after the final fallback.

### Candidate traversal and training semantics

Each 32K training pool has a model-independent, seeded candidate-order
manifest. A formal single-rank run follows this exact permutation without
replacement or reshuffling: 32,000 unique rows / 16 candidates = exactly 2,000
logical optimizer steps. Formal preflight fixes one epoch, batch 16,
`drop_last=true`, and gradient accumulation 1; `--max-steps < 2000` is allowed
only as a smoke-prefix run.

The Dr.Post-Training algorithm is unchanged. `FullTraining` updates on all 16
candidates; hard-selection methods retain `k=8`; `LayerwiseSoft` and
`LayerwiseSoftP` retain their existing continuous weighting semantics and do
not round to a hard subset. Target-gradient batches remain deterministic
two-example logical batches, and target validation is never used for selection.

The following commands exercise three steps without submitting a full job:

```bash
bash SFT/train/train.sh -c configs/dolci32k/inst_if \
  -m FullTraining --model-profile qwen3_1_7b --max-steps 3

bash SFT/train/train.sh -c configs/dolci32k/inst_if \
  -m FullTraining --model-profile qwen3_4b --max-steps 3

bash SFT/train/train.sh -c configs/dolci32k/inst_if \
  -m FullTraining --model-profile qwen3_8b --max-steps 3
```

Inspect the formal five-setting campaign without submitting it, then omit
`--dry-run` to launch:

```bash
bash SFT/train/submit_general_loss_comparison.sh \
  --profile dolci32k --model-profile qwen3_1_7b \
  --campaign-id <unique-id> --dry-run

bash SFT/train/submit_general_loss_comparison.sh \
  --profile dolci32k --model-profile qwen3_1_7b \
  --campaign-id <unique-id>
```

Substitute `olmo3_7b`, `qwen3_4b`, or `qwen3_8b` for a separate model campaign. Training is
intentionally single-rank because sharding the candidate window would change
the algorithm. The OLMo-7B and Qwen-8B overlays and launch interfaces are
supported, but their full-parameter optimizer-state floors leave no viable
activation headroom on one 48 GB A40. Treat those profiles as interface
support, not one-A40 campaign support; use a larger-memory device for full
training.

Run the lightweight contract tests before a campaign:

```bash
python -m pytest tests/test_dolci32k_data.py tests/test_dolci32k_window.py
```

### Target-gradient signal (`--target-signal`)

Curation scores a candidate by how well its gradient aligns with a target
gradient built from D*, the 64 rows in `targets/<target>/grad.jsonl`. That
target gradient has always been the gradient of one objective: token-mean
cross entropy on the reference trajectory. On the math and code settings the
methods lower target validation loss and still lose on MATH500 / MBPP+, and
target-only SFT (`SFT/train/train_target_only.py`) reproduces that split, so
the objective itself is a suspect. `--target-signal` varies exactly that
objective and nothing else -- same build, same pool, same traversal, same
optimizer -- so a downstream difference cannot be attributed to anything else.

| Signal | Target gradient is the gradient of | Extra artifacts |
| --- | --- | --- |
| `nll` (default) | token-mean CE over the whole reference trajectory | none |
| `answer_only_ce` | the same CE, restricted to final-answer tokens | none |
| `correct_incorrect_margin` | `softplus(beta * (margin - (logp_correct - logp_incorrect)))` over per-response mean log probabilities | candidates |
| `reward_weighted_sft` | correctness-weighted token mean `sum r*NLL / sum r*tokens` | candidates |

`nll` is untouched: it keeps its own loader, collator, and loss call, so the
control arm is the same code that produced the existing results.

**Final-answer spans.** `answer_only_ce` supervises the last `\boxed{...}` for
`math` and the last fenced code block for `mbpp` -- about 9% and 93% of the
reference tokens respectively. `precise_if` has no separable final answer: the
response *is* the answer, so the span is the whole assistant turn and the
signal is numerically identical to `nll` there. The run logs that and records
`answer_span_degenerate` in its metadata rather than implying a distinction
that does not exist.

**Candidates are generated once, offline.** Neither grouped signal adds a
rollout, a reward model, or an online RL step to the training loop; both read a
jsonl written ahead of time by a pinned Qwen3 generator whose samples are
scored by the domain's verifier -- Math-Verify for `math`, the row's own
`test_list` executed in a subprocess for `mbpp`, and IFEval constraints
recovered from the prompt and checked with the vendored official verifier for
`precise_if`. Recovery for `precise_if` is deliberately lenient (an
unrecognized constraint family is not checked) and any prompt whose own gold
reference fails the recovered constraints is dropped, so the verifier can miss
a violation but cannot invent one.

```bash
# Once per (build, generator). ~1 GPU-hour for all three targets.
python -m SFT.data.build_target_candidates \
  --artifact_build_id "$(cat SFT/data/dolci32k_artifacts/CURRENT)" \
  --generator_profile qwen3_4b --num_samples 6 --temperature 1.0

# Or as a 3-task Slurm array:
DRPT_ARTIFACT_BUILD_ID="$(cat SFT/data/dolci32k_artifacts/CURRENT)" \
  sbatch SFT/data/build_target_candidates_job.sh
```

Output lands beside the build, never inside it -- the build directory is
content addressed, so adding a file there would invalidate its own id:

```
SFT/data/dolci32k_artifacts/target_signals/<build_id>/<target>/candidates.jsonl
SFT/data/dolci32k_artifacts/target_signals/<build_id>/<target>/manifest.json
```

The manifest records the generator pin, sampling settings, and a verification
histogram. Check `counts.margin_pair_coverage` before launching a margin run:
it is the fraction of the 64 prompts that produced at least one verified-wrong
trajectory, and it is the number of pairs the margin loss actually has.

**Coverage below 64 is not cosmetic.** The margin objective drops a prompt with
no negative, so its D* becomes a subset of the 64 rows its `nll` control sees,
and a downstream difference could then be caused by the smaller prompt set
rather than by the objective. Two ways to remove that confound:

```bash
# 1. Top up: re-sample only the prompts that came out all-correct and merge.
#    Idempotent -- prompts that already have a negative are skipped.
python -m SFT.data.build_target_candidates \
  --artifact_build_id "$(cat SFT/data/dolci32k_artifacts/CURRENT)" \
  --only_missing_negatives --num_samples 24 --temperature 1.15
```

Use a *moderate* temperature increase. Pushing high enough to make a strong
generator fail an easy prompt also produces collapsed output, and a negative
that is a few mojibake characters teaches the margin loss to reject noise
rather than wrong reasoning. Such trajectories are still written to the
artifact but are excluded from selection by `MIN_INFORMATIVE_NEGATIVE_CHARS`,
so a prompt whose only negative is noise correctly reports as having none.

Option 2, for coverage that cannot be closed — `precise_if` has prompts whose
constraints cannot be recovered at all, so it is capped below 64 by
construction — is `--target-signal-align-prompts`. It restricts **every**
signal, `nll` included, to the margin-usable subset, making the target prompt
set identical across all four arms. Pass it to every arm of a comparison or to
none; aligned runs get an `-aligned` suffix so they never overwrite an
unaligned control.

Then run any signal through the ordinary entry point. A non-default signal
appends `-ts<signal>` to the run name, so it never overwrites its `nll` control:

```bash
bash SFT/train/train.sh -c configs/dolci32k/reason_math \
  -m LayerWiseSubset --model-profile qwen3_1_7b \
  --target-signal answer_only_ce --max-steps 3

# 3 settings x 4 signals, setting-major:
DRPT_CAMPAIGN_ID=<unique-id> \
DRPT_ARTIFACT_BUILD_ID="$(cat SFT/data/dolci32k_artifacts/CURRENT)" \
  sbatch SFT/train/target_signal_comparison_job.sh
```

Knobs: `--target-signal-beta` and `--target-signal-margin` shape the margin
penalty; `--target-signal-incorrect-reward` (default `0.0`, i.e. rejection
sampling fine-tuning) keeps wrong trajectories at reduced weight under
`reward_weighted_sft`.

Target microbatching stays exact under every signal -- the scaled chunk losses
sum to the loss one unchunked backward would have produced -- but the chunk
unit differs. `reward_weighted_sft` is still a weighted token mean with no
coupling between rows, so it chunks by rows under the usual
`target_microbatch_size`. `correct_incorrect_margin` compares two trajectories
of the same prompt, so splitting a pair across chunks would change the loss;
it chunks by whole prompt groups under `target_signal_groups_per_microbatch`
(default 1), which `target_microbatch_size` cannot express.

```bash
python -m pytest tests/test_target_gradient_signals.py tests/test_target_signal_pipeline.py
```

## Dolci32k campaign resources and evaluation

The canonical matrix is setting-major and method-minor: AdamW indices 0-24
(5 settings x 5 methods), followed by Muon logical indices 0-39 (5 x 8).
Formal launches require a three-step smoke gate. AdamW uses LR `1e-5`; Muon
uses matrix LR `3e-4` and auxiliary AdamW LR `1e-5`.

The logical algorithm is always N=16 candidates, k=8 selected mass, and T=2
target examples. Candidate and target microbatches only chunk that computation;
the default candidate C=2 path still makes one 16-way decision and one outer
optimizer step. Gradient checkpointing is enabled with non-reentrant replay.
The guarded C=8/C=16 probe remains a memory/runtime experiment, not a change to
the production algorithm.

Soft and SoftP keep their continuous solver/weight state in FP32. Their final
large replay contraction uses BF16 operands with FP32 accumulation/output
(`soft_weighting.replay_precision: bf16_fp32`) to reduce runtime while
preserving the feasible set and weighting rule.

Host-memory defaults are 48 GB for AdamW and light Muon cells and 192 GB for
Muon Soft/SoftP cells. Override them with `DRPT_DOLCI_ADAMW_MEM`,
`DRPT_DOLCI_MUON_LIGHT_MEM`, and `DRPT_DOLCI_MUON_HEAVY_MEM` after checking
the smoke run peak-memory metadata.

A complete campaign automatically submits downstream evaluation after each
family reaches a terminal state. AdamW has 35 benchmark cells and Muon has 56:
IFEval and IFBench are separate rows, while MATH500 and MBPP+ each contribute
one row for their settings. Only complete, provenance-matched checkpoints and
full official task sets enter the collected report.

## Legacy multi-finetuning scope (4 active settings)

3 LoRA-only train→target settings + 1 multi-finetuning setting
(`alpaca → samsum`, covering Full/LoRA/MeSO), each at 5 seeds. Per-task
target-only baselines train directly on `n_val=16` validation samples.

| # | Config dir       | Train pool | Target task | Step budget | `eval_steps` | Methods                  |
|---|------------------|------------|-------------|-------------|--------------|--------------------------|
| 1 | `alpaca_samsum`  | alpaca     | samsum      | 2600        | 26           | 9 (Full+LoRA+MeSO × 3 curations) |
| 2 | `less_tydiqa`    | less mix   | tydiqa      | 1225        | 12           | 3 (LoRA × 3 curations)   |
| 3 | `triviaqa_nq`    | triviaqa   | nq_open     | 1107        | 11           | 3 (LoRA × 3 curations)   |
| 4 | `less_squad`     | less mix   | squad       | 1225        | 12           | 3 (LoRA × 3 curations)   |

LESS mix = `flan_v2 + cot + dolly + oasst1` (~1.96M). Run-dir prefix is
`{train}_{task}` so setting 3 produces `triviaqa_nq_open-...`.

## Hyperparameters

Fixed across all settings. No LR tuning per setting.

| Setting | Value |
|---|---|
| Model | `meta-llama/Llama-3.2-1B` |
| LR (Full / MeSO) | `1e-5` |
| LR (LoRA) | `1e-4` |
| Scheduler | linear, `warmup_ratio=0.03` |
| Optimizer | AdamW for the legacy sweep; AdamW or official-first Muon with auxiliary AdamW for the focused comparison (`weight_decay=0.0`) |
| Precision | bf16, flash-attention-2 |
| LoRA | `r=8`, `alpha=16`, `dropout=0.1`, `target_modules=all-linear` |
| Batch size | `per_device=8`, `gradient_accumulation=1` |
| Seq length | `max_seq_length=512` |
| Curation | `selection_frac=0.5`, `n_val=16`, `val_strategy=merged_batch`, `scoring.method=reduced_ghost` (LayerWiseSubset uses `compress` with `compression=normal-64*64`) |
| MeSO | optimizer `compression=normal-512*512` |
| Eval | `n_eval=500`, `n_test=500`, seeds {2, 22, 42, 62, 82} |

## Chat template

All examples are stored as `messages` JSONL (no template baked in).
Llama-3.2-1B-Base ships without a chat template, so we install an
open-instruct-style fallback (`<|user|>` / `<|assistant|>` plaintext
markers) via `SFT/data/get_val_dataset.py:ensure_chat_template`. Both
training and eval call `tokenizer.apply_chat_template(...)` with this
template; loss is computed only on the assistant-content tokens.

## Data preparation

```bash
# Eval splits (val/lr/test) for the 4 active target tasks
python SFT/data/prepare_datasets.py --datasets samsum tydiqa nq_open_eval squad_eval

# Training pools
python SFT/data/prepare_datasets.py --datasets alpaca triviaqa_train dolly oasst1 flan_v2 cot
```

`cot` (`kaist-ai/CoT-Collection`) is loaded via
`revision="refs/convert/parquet"` because the script form is rejected by
`datasets >= 3.0`.

| Dataset    | Role  | Lines (post-prep)        | Description                                         |
| ---------- | ----- | ------------------------ | --------------------------------------------------- |
| `samsum`   | eval  | 818 / 100 / 719          | Dialogue summarization (val/lr/test)                |
| `tydiqa`   | eval  | 100 / 100 / 4877         | Multilingual extractive QA (val/lr/test)            |
| `nq_open`  | eval  | val/lr/test from HF validation (~3.6K) | Closed-book factoid QA               |
| `squad`    | eval  | val/lr/test from HF validation         | Closed-book reading-comprehension QA |
| `alpaca`   | train | 52,002                   | Stanford Alpaca instruction-following               |
| `triviaqa` | train | ~138K                    | TriviaQA closed-book Q→A pairs (rc.nocontext)       |
| `flan_v2`  | train | 100,000 (subset)         | LESS-mix component                                  |
| `cot`      | train | 1,837,928                | LESS-mix component (CoT-Collection, parquet rev.)   |
| `dolly`    | train | 15,011                   | LESS-mix component                                  |
| `oasst1`   | train | 9,846                    | LESS-mix component (multi-turn unrolled)            |

## Methods (per setting)

| Config                  | Curation       | Finetuning |
|-------------------------|----------------|------------|
| `FullTraining-Full`     | none           | Full       |
| `FullTraining-LoRA`     | none           | LoRA r=8   |
| `FullTraining-MeSO`     | none           | MeSO       |
| `LayerWiseSubset-Full`  | per-layer top-k| Full       |
| `LayerWiseSubset-LoRA`  | per-layer top-k| LoRA r=8   |
| `LayerWiseSubset-MeSO`  | per-layer top-k| MeSO       |
| `GlobalSubset-Full`     | global top-k   | Full       |
| `GlobalSubset-LoRA`     | global top-k   | LoRA r=8   |
| `GlobalSubset-MeSO`     | global top-k   | MeSO       |

Setting 1 (`alpaca_samsum`) runs all 9; settings 2–4 run only the 3 LoRA
variants. Per-task target-only baselines (`FullTraining-{Full,LoRA,MeSO}`
via `train_val_ablation.sh`) train directly on the `n_val=16` task
validation samples.

> Run dirs: `{train}_{task}-{model}-{Method}-p{pct}-lr{lr}-b{batch}-v{nval}-s{seed}`

## Submitting the legacy full sweep

```bash
# 90 main + 30 target-only + 18 eval-main + 6 eval-target = 144 jobs
bash SFT/train/submit_all.sh             # submit
bash SFT/train/submit_all.sh --dry-run   # print sbatch commands only
```

Layout:
- Stage 1: 90 main training jobs (3h walltime)
- Stage 2: 30 target-only jobs (2h walltime)
- Stage 3: 18 main-eval jobs (2h, depends on Stage 1)
- Stage 4: 6 target-eval jobs (2h, depends on Stage 2)

## Single-job training

```bash
bash SFT/train/train.sh -c configs/<setting> -m all
bash SFT/train/train.sh -c configs/<setting> -m FullTraining-Full --seed 42
bash SFT/train/train.sh -c configs/<setting> --list
```

Categories: `all`, `full-training`, `layer-wise-subset`, `global-subset`,
`full`, `lora`, `meso`.

```bash
bash SFT/train/train_val_ablation.sh \
    --task <target_task> --config_dir <setting> \
    --methods FullTraining-Full --eval_steps <n> --seed <seed>
```

## Evaluation

```bash
# n_test=500 matches the during-training perplexity sample for direct comparison
bash SFT/eval/eval.sh --train <train> --task <task> --batch_size 64 --n_test 500
```

Supported tasks: `samsum`, `tydiqa`, `nq_open`, `squad`, `triviaqa`, plus the
dolci32k benchmarks `ifeval`, `ifbench`, `math500`, `mbpp_plus`.

`evaluate` and `rouge_score` Python packages must be installed in the
active env (`pip install evaluate rouge_score`).

## Config directory structure

Each config dir has `defaults.yaml` (shared) and one YAML per method:

```
configs/<setting>/
  defaults.yaml              # model, dataset, scheduler, etc.
  FullTraining-{Full,LoRA,MeSO}.yaml
  GlobalSubset-{Full,LoRA,MeSO}.yaml
  LayerWiseSubset-{Full,LoRA,MeSO}.yaml
```

`defaults.yaml`:
```yaml
model: meta-llama/Llama-3.2-1B
train_dataset: <pool>
target_task: <task>
percentage: <pct>

seed: 42
batch_size: 8
gradient_accumulation_steps: 1
optim: adamw_torch
max_seq_length: 512
lr_scheduler_type: linear
warmup_ratio: 0.03
weight_decay: 0.0
num_train_epochs: 1
eval_steps: <n>          # ~100 ppl points across max_steps
use_flash_attention: true

n_eval: 500
selection_frac: 0.5
selection_mode: topk
n_val: 16
val_batch_size: 1
val_strategy: merged_batch
scoring:
  method: reduced_ghost
```

Load order: defaults → `defaults.yaml` → method YAML → CLI (`--seed`, `--lr`).

#### Adding a new setting

1. Create `configs/<new_setting>/` with a `defaults.yaml`.
2. Copy method YAMLs (3 if LoRA-only, 9 if Full+LoRA+MeSO) — LRs are fixed (`1e-5` / `1e-4`).
3. Prep data: `python SFT/data/prepare_datasets.py --datasets <pool> <task>`.
4. Add the setting (and any new target task) to `submit_all.sh`.
