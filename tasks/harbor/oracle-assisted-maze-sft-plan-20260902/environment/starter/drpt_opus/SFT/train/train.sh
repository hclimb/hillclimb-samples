#!/bin/bash
#
# SFT Training Runner
#
# All experiment settings live in config files.
# Each config directory has defaults.yaml (shared settings) + per-method configs.
#
# Usage: bash train.sh -c <config_dir> -m <methods> [options]
#

# An explicit repository root wins over the submission directory.  This keeps
# spooled Slurm jobs correct even when sbatch was invoked outside the checkout.
REPO_ROOT="${DRPT_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
source "$REPO_ROOT/cluster_env.sh" \
    || { echo "ERROR: $REPO_ROOT/cluster_env.sh not found."; exit 1; }
activate_env

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

SCRIPT_DIR="$REPO_ROOT/SFT/train"

# Fixed training args (infra-level, always the same).  TF32 remains disabled by
# default for baseline comparability; exceptionally expensive runs may opt in
# through the job environment, which is captured by TrainingArguments/W&B.
TF32_VALUE="${TF32:-False}"
if [[ "$TF32_VALUE" != "True" && "$TF32_VALUE" != "False" ]]; then
    echo "ERROR: TF32 must be exactly True or False, got: $TF32_VALUE"
    exit 1
fi
FIXED_ARGS="--do_train=True \
--do_eval=True \
--use_fast_tokenizer=True \
--logging_steps=1 \
--eval_strategy=steps \
--save_strategy=no \
--bf16=True \
--fp16=False \
--overwrite_output_dir=True"

# =============================================================================
# CLI
# =============================================================================
config_dir=""
methods=""
seed_override=""
lr_override=""
muon_lr_override=""
aux_adamw_lr_override=""
optimizer_type_override=""
eval_split_override=""
report_to_override=""
wandb_project_override=""
wandb_run_name_override=""
wandb_group_override=""
wandb_tags_override=""
campaign_id_override=""
artifact_build_id_override=""
model_profile_override=""
max_steps_override=""
candidate_microbatch_override=""
target_signal_override=""
target_signal_beta_override=""
target_signal_margin_override=""
target_signal_incorrect_reward_override=""
target_signal_align_prompts_override=""
allow_dolci_c16_probe=false
retry_failed=false
dry_run=false

SFT10_METHODS=(
    FullTraining
    GlobalRaw
    LayerwiseRaw
    OptGroupRaw
    GlobalOptA
    LayerwiseOptA
    OptGroupOptA
    GlobalOptB
    LayerwiseOptB
    OptGroupOptB
)

NEW_SOLVER_METHODS=(
    GlobalRandom
    LayerwiseRandom
    GlobalSoft
    LayerwiseSoft
    LayerwiseSoftP
    GlobalHybridMuonSur
    LayerwiseHybridMuonSur
    GlobalHybridMuonMatrixSur
    LayerwiseHybridMuonMatrixSur
    GlobalMuonSur
    LayerwiseMuonSur
    LayerwiseMuonPSur
    LayerwiseMuonSatSur
    LayerwiseMuonSatPSur
)

ADAMW_CORE_METHODS=(
    FullTraining
    LayerwiseRaw
    LayerwiseSoft
    LayerwiseSoftP
    LayerwiseOptA
)

MUON_CORE_METHODS=(
    FullTraining
    LayerwiseRaw
    LayerwiseSoft
    LayerwiseSoftP
    LayerwiseMuonSur
    LayerwiseMuonPSur
    LayerwiseMuonSatSur
    LayerwiseMuonSatPSur
)

HYBRID_CORE_METHODS=(
    FullTraining
    GlobalRaw
    LayerwiseRaw
    LayerwiseHybridMuonSur
    GlobalSoft
    LayerwiseSoft
)

MUON_SURROGATE_VARIANT_METHODS=(
    LayerwiseMuonSur
    LayerwiseMuonPSur
    LayerwiseMuonSatSur
    LayerwiseMuonSatPSur
)

SOFT_VARIANT_METHODS=(
    LayerwiseSoft
    LayerwiseSoftP
)

while [[ $# -gt 0 ]]; do
    case $1 in
        --config_dir|-c)  config_dir="$2"; shift 2 ;;
        --methods|-m)     methods="$2"; shift 2 ;;
        --seed)           seed_override="$2"; shift 2 ;;
        --lr)             lr_override="$2"; shift 2 ;;
        --muon_lr|--muon-lr) muon_lr_override="$2"; shift 2 ;;
        --aux_adamw_lr|--aux-adamw-lr) aux_adamw_lr_override="$2"; shift 2 ;;
        --optimizer_type) optimizer_type_override="$2"; shift 2 ;;
        --eval_split)     eval_split_override="$2"; shift 2 ;;
        --report_to)      report_to_override="$2"; shift 2 ;;
        --wandb_project)  wandb_project_override="$2"; shift 2 ;;
        --wandb_run_name) wandb_run_name_override="$2"; shift 2 ;;
        --wandb_group)    wandb_group_override="$2"; shift 2 ;;
        --wandb_tags)     wandb_tags_override="$2"; shift 2 ;;
        --campaign_id|--campaign-id) campaign_id_override="$2"; shift 2 ;;
        --artifact_build_id|--artifact-build-id) artifact_build_id_override="$2"; shift 2 ;;
        --model-profile|--model_profile) model_profile_override="$2"; shift 2 ;;
        --max_steps|--max-steps) max_steps_override="$2"; shift 2 ;;
        --candidate_microbatch_size|--candidate-microbatch-size) candidate_microbatch_override="$2"; shift 2 ;;
        --target_signal|--target-signal) target_signal_override="$2"; shift 2 ;;
        --target_signal_beta|--target-signal-beta) target_signal_beta_override="$2"; shift 2 ;;
        --target_signal_margin|--target-signal-margin) target_signal_margin_override="$2"; shift 2 ;;
        --target_signal_incorrect_reward|--target-signal-incorrect-reward) target_signal_incorrect_reward_override="$2"; shift 2 ;;
        --target_signal_align_prompts|--target-signal-align-prompts) target_signal_align_prompts_override="true"; shift ;;
        --allow-dolci-c16-probe) allow_dolci_c16_probe=true; shift ;;
        --retry-failed)  retry_failed=true; shift ;;
        --dry-run)        dry_run=true; shift ;;
        --list)
            dir="${config_dir:-configs}"
            [[ "$dir" != /* ]] && dir="$SCRIPT_DIR/$dir"
            echo "Available methods in $dir:"
            for f in "$dir"/*.yaml; do
                [[ ! -f "$f" ]] && continue
                name=$(basename "$f" .yaml)
                [[ "$name" != "defaults" ]] && echo "  $name"
            done
            if [[ -f "$dir/OptimizerAwareGlobalSubset-Full.yaml" ]]; then
                echo ""
                echo "sft10 method labels:"
                for m in "${SFT10_METHODS[@]}"; do
                    echo "  $m"
                done
            fi
            echo ""
            echo "Categories: all, sft10, baseline9-adamw, baseline9-muon, adamw-core, muon-core, hybrid-core, soft-variants, muon-surrogate-variants, new-solvers, full-training, layer-wise-subset, optimizer-aware, optimizer-group-wise, global-subset, opus-baselines, full, lora, meso"
            exit 0
            ;;
        --help|-h)
            cat <<'HELP'
Usage: bash train.sh -c <config_dir> -m <methods> [options]

All experiment settings (model, batch_size, dataset, LR, etc.) live in config files.
Each config directory has a defaults.yaml for shared settings, plus per-method configs.

Required:
  -c, --config_dir <dir>  Config directory (relative to SFT/train/ or absolute)
  -m, --methods <list>    Methods or categories (comma-separated)

Optional:
  --seed <seed>           Override seed from config
  --lr <lr>               Override learning rate from config
  --muon-lr <lr>          Override Muon-managed matrix learning rate
  --aux-adamw-lr <lr>     Override auxiliary AdamW learning rate
  --optimizer_type <type> Override optimizer.type ("adamw", "muon", or "hybrid")
  --eval_split <split>    Override eval split ("test" or "lr")
  --report_to <target>    Logging target passed to HF Trainer ("wandb" or "none")
  --wandb_project <name>  Weights & Biases project name
  --wandb_run_name <name> Weights & Biases run name
  --wandb_group <name>    Weights & Biases group name
  --wandb_tags <tags>     Comma-separated Weights & Biases tags
  --campaign_id <id>      Store under runs/campaigns/<id> without overwriting legacy runs
  --artifact-build-id ID  Pin an audited profile artifact build (normally set by launcher)
  --model-profile NAME    dolci32k model overlay: olmo3_7b, qwen3_1_7b, qwen3_4b, or qwen3_8b
  --max_steps <count>     Optional Trainer max-steps override (primarily for smoke tests)
  --candidate-microbatch-size N
                          Override dolci32k candidate micro-batch size
  --allow-dolci-c16-probe Permit C=16 only for an explicit max_steps<=3 probe
  --retry-failed          Retry an incomplete/failed dolci32k run; completed runs stay protected
  --dry-run               Print commands without executing
  --list                  List available methods and exit

Categories: all, sft10, baseline9-adamw, baseline9-muon, adamw-core, muon-core, hybrid-core, soft-variants, muon-surrogate-variants, new-solvers, full-training, layer-wise-subset, optimizer-aware, optimizer-group-wise, global-subset, opus-baselines, full, lora, meso

Examples:
  bash train.sh -c configs/alpaca_samsum -m all
  bash train.sh -c configs/alpaca_samsum -m adamw-core --dry-run
  bash train.sh -c configs/alpaca_samsum -m muon-core --dry-run
  bash train.sh -c configs/alpaca_samsum -m hybrid-core --dry-run
  bash SFT/train/submit_general_loss_comparison.sh --campaign-id <ID>  # baseline9: AdamW 5 + Muon 8 x four tasks
  bash train.sh -c configs/less_tydiqa -m "LayerWiseSubset-LoRA,GlobalSubset-LoRA" --seed 123
  bash train.sh -c configs/alpaca_samsum -m full-training --dry-run
HELP
            exit 0
            ;;
        *) echo "Unknown argument: $1 (use --help)"; exit 1 ;;
    esac
done

# Validate required args
if [[ -z "$config_dir" ]] || [[ -z "$methods" ]]; then
    echo "Usage: bash train.sh -c <config_dir> -m <methods> [options]"
    echo "       bash train.sh --help"
    exit 1
fi

if [[ -n "$campaign_id_override" && ! "$campaign_id_override" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "ERROR: campaign_id must contain only letters, digits, '.', '_' or '-' and must start alphanumeric"
    exit 1
fi
if [[ -n "$max_steps_override" && ( ! "$max_steps_override" =~ ^[1-9][0-9]*$ ) ]]; then
    echo "ERROR: max_steps must be a positive integer"
    exit 1
fi
if [[ -n "$candidate_microbatch_override" && ( ! "$candidate_microbatch_override" =~ ^[1-9][0-9]*$ ) ]]; then
    echo "ERROR: candidate_microbatch_size must be a positive integer"
    exit 1
fi
if [[ -n "$artifact_build_id_override" \
      && ! "$artifact_build_id_override" =~ ^[0-9a-f]+$ \
      && ! ( "$dry_run" == "true" && "$artifact_build_id_override" == "CURRENT" ) ]]; then
    echo "ERROR: artifact_build_id must contain only lowercase hexadecimal characters"
    exit 1
fi

# Focused categories intentionally choose one optimizer family.  A Muon run
# uses official torch.optim.Muon for eligible matrices and an auxiliary AdamW
# optimizer for parameter shapes that official Muon does not accept.
case "$methods" in
    solver-comparison|focused-solver-comparison)
        echo "ERROR: solver-comparison spans two optimizer families and cannot be " \
             "represented by one train.sh method list." >&2
        echo "Use: bash SFT/train/submit_general_loss_comparison.sh --campaign-id <ID>" >&2
        exit 2
        ;;
    adamw-core|focused-adamw|baseline9-adamw)
        if [[ -n "$optimizer_type_override" && "$optimizer_type_override" != "adamw" ]]; then
            echo "ERROR: $methods is AdamW-only; got --optimizer_type $optimizer_type_override"
            exit 1
        fi
        optimizer_type_override="adamw"
        ;;
    muon-core|focused-muon|baseline9-muon)
        if [[ -n "$optimizer_type_override" && "$optimizer_type_override" != "muon" ]]; then
            echo "ERROR: $methods requires --optimizer_type muon"
            exit 1
        fi
        optimizer_type_override="muon"
        ;;
    hybrid-core|focused-hybrid)
        if [[ -n "$optimizer_type_override" && "$optimizer_type_override" != "hybrid" ]]; then
            echo "ERROR: $methods requires --optimizer_type hybrid"
            exit 1
        fi
        optimizer_type_override="hybrid"
        ;;
esac

# Resolve config dir to absolute path
[[ "$config_dir" != /* ]] && config_dir="$SCRIPT_DIR/$config_dir"

if [[ ! -d "$config_dir" ]]; then
    echo "ERROR: Config directory not found: $config_dir"
    exit 1
fi

# A setting may delegate its method YAMLs to a shared directory via
# `method_config_dir:` in defaults.yaml. Method resolution runs before the full
# YAML parse, so read just that one key up front.
shared_method_dir=""
if [[ -f "$config_dir/defaults.yaml" ]]; then
    shared_method_dir="$(grep -E '^method_config_dir:[[:space:]]*' "$config_dir/defaults.yaml" \
        | head -1 | cut -d: -f2- | xargs)"
    if [[ -n "$shared_method_dir" ]]; then
        [[ "$shared_method_dir" != /* ]] && shared_method_dir="$SCRIPT_DIR/$shared_method_dir"
        if [[ ! -d "$shared_method_dir" ]]; then
            echo "ERROR: method_config_dir not found: $shared_method_dir"
            exit 1
        fi
    fi
fi

# Path of a method YAML, preferring the setting's own directory over the shared
# one. Prints nothing and returns 1 when neither has it.
method_config_file() {
    local stem="$1"
    if [[ -f "$config_dir/${stem}.yaml" ]]; then
        echo "$config_dir/${stem}.yaml"
        return 0
    fi
    if [[ -n "$shared_method_dir" && -f "$shared_method_dir/${stem}.yaml" ]]; then
        echo "$shared_method_dir/${stem}.yaml"
        return 0
    fi
    return 1
}

profile_hint=""
if [[ -f "$config_dir/defaults.yaml" ]]; then
    profile_hint="$(grep -E '^experiment_profile:[[:space:]]*' "$config_dir/defaults.yaml" \
        | head -1 | cut -d: -f2- | xargs)"
fi

dolci_artifact_root=""
if [[ "$profile_hint" == "dolci32k" ]]; then
    if ! dolci_artifact_root="$(
        "$DRPT_PYTHON" "$REPO_ROOT/SFT/data/prepare_dolci32k.py" \
            --data-dir "$DRPT_DATA_DIR" --print-artifact-root
    )" || [[ "$dolci_artifact_root" != /* || "$dolci_artifact_root" == *$'\n'* ]]; then
        echo "ERROR: failed to resolve the canonical dolci32k artifact root" >&2
        exit 1
    fi
fi

if [[ -n "$model_profile_override" && "$profile_hint" != "dolci32k" ]]; then
    echo "ERROR: --model-profile is supported only by the dolci32k profile" >&2
    exit 2
fi

if [[ "$dry_run" != "true" ]]; then
    if [[ "$profile_hint" == "dolci32k" ]]; then
        dolci_audit_args=(--data-dir "$DRPT_DATA_DIR" --audit-only)
        pinned_build_id="${artifact_build_id_override:-${DRPT_ARTIFACT_BUILD_ID:-}}"
        [[ -n "$pinned_build_id" ]] && dolci_audit_args+=(--build-id "$pinned_build_id")
        if ! "$DRPT_PYTHON" "$REPO_ROOT/SFT/data/prepare_dolci32k.py" "${dolci_audit_args[@]}"; then
            echo "ERROR: pinned dolci32k artifacts are missing, stale, or failed audit." >&2
            echo "Build first with: $DRPT_PYTHON $REPO_ROOT/SFT/data/prepare_dolci32k.py --data-dir $DRPT_DATA_DIR --build" >&2
            exit 1
        fi
    else
        case "$(basename "$config_dir")" in
        alpaca_samsum|less_squad|less_tydiqa|triviaqa_nq)
            if ! "$DRPT_PYTHON" "$REPO_ROOT/SFT/data/prepare_baseline9.py" \
                --data-dir "$DRPT_DATA_DIR" --check-only; then
                echo "ERROR: baseline9 data are incomplete." >&2
                echo "Run: $DRPT_PYTHON $REPO_ROOT/SFT/data/prepare_baseline9.py --data-dir $DRPT_DATA_DIR" >&2
                exit 1
            fi
            ;;
        esac
    fi
fi

# =============================================================================
# Config parser
# =============================================================================
reset_config() {
    # Method
    cfg_method="FullTraining"
    cfg_finetuning="Full"

    # Scoring (nested under scoring: in YAML)
    cfg_scoring_method="reduced_ghost"
    cfg_score_compression=""     # Single string: "normal-64*64" or "normal-64*64/sjlt-512"

    # Optimizer compression (nested under optimizer: in YAML)
    cfg_optimizer_type="adamw"
    cfg_muon_learning_rate=""
    cfg_aux_adamw_learning_rate=""
    cfg_opt_compression=""       # Single string: "normal-512*512" or "normal-512*512/sjlt-256"

    # Curation
    cfg_selection_frac="0.5"
    cfg_selection_mode="topk"
    cfg_n_val="8"
    cfg_val_batch_size="1"
    cfg_val_strategy="merged_batch"
    cfg_use_second_order="false"
    cfg_subset_mode="one_pass"

    # Optimizer-aware group-wise curation
    cfg_oa_matrix_geometry="muon"
    cfg_oa_vector_geometry="adamw"
    cfg_oa_target_mode="opta"
    cfg_oa_adam_eps="1e-8"
    cfg_oa_muon_reference="momentum_proxy"
    cfg_oa_muon_momentum="0.95"
    cfg_oa_muon_nesterov="true"
    cfg_oa_muon_steps="5"
    cfg_oa_muon_eps="1e-7"
    cfg_oa_muon_max_dim="256"
    cfg_oa_muon_lr_shape_scale="true"
    cfg_oa_muon_adjust_lr_fn="original"
    cfg_oa_muon_backend="auto"
    cfg_oa_lora_optimizer="adamw"
    cfg_oa_spectral_lambda="0.0"
    cfg_oa_spectral_eps="1e-12"
    cfg_oa_token_normalized_selection="false"

    # Continuous soft-weighting solver
    cfg_soft_steps="20"
    cfg_soft_lr="0.1"
    cfg_soft_tol="1e-5"
    cfg_soft_patience="3"
    cfg_soft_gamma="0.0"
    cfg_soft_use_optimizer_state="true"
    cfg_soft_constraint="capped_simplex"
    cfg_soft_replay_precision="fp32"

    # Muon spectral-support surrogate
    cfg_muon_sur_alpha="1.0"
    cfg_muon_sur_include_adamw_scores="true"
    cfg_muon_sur_mode_weighting="uniform"
    cfg_muon_sur_saturation="false"
    cfg_muon_sur_rank="32"
    cfg_muon_sur_full_svd_max_dim="256"
    cfg_muon_sur_rtol="1e-6"
    cfg_muon_sur_oversample="8"
    cfg_muon_sur_power_iters="2"

    # LoRA
    cfg_lora_r="8"
    cfg_lora_alpha="16"
    cfg_lora_dropout="0.1"

    # Experiment
    cfg_experiment_profile=""
    cfg_setting_id=""
    cfg_artifact_build_id="${DRPT_ARTIFACT_BUILD_ID:-}"
    cfg_model_profile=""
    cfg_model="meta-llama/Llama-3.2-1B"
    cfg_model_revision="main"
    cfg_tokenizer=""
    cfg_tokenizer_revision=""
    cfg_seed="42"
    cfg_batch_size="8"
    cfg_gradient_accumulation_steps="1"
    cfg_n_eval="500"
    cfg_n_target_val=""
    cfg_optim="adamw_torch"
    cfg_use_flash_attention="true"
    cfg_gradient_checkpointing="false"
    cfg_tf32="$TF32_VALUE"
    cfg_dataloader_drop_last="false"
    cfg_learning_rate=""

    # Exact logical-window engine. Empty values preserve legacy execution.
    cfg_logical_candidate_batch_size=""
    cfg_candidate_microbatch_size=""
    cfg_target_microbatch_size=""
    cfg_target_cache_device="cuda"
    cfg_target_cache_pin_memory="false"

    # Objective whose gradient defines the target signal. "nll" is the historical
    # default and reads nothing beyond the immutable build; the others consume the
    # offline candidates written by SFT/data/build_target_candidates.py.
    cfg_target_signal_mode="nll"
    cfg_target_signal_beta="1.0"
    cfg_target_signal_margin="0.0"
    cfg_target_signal_incorrect_reward="0.0"
    cfg_target_signal_max_candidates=""
    cfg_target_signal_groups_per_microbatch=""
    cfg_target_signal_align_prompts="false"

    # Training hyperparameters
    cfg_max_seq_length="512"
    cfg_lr_scheduler_type="cosine"
    cfg_warmup_ratio="0.1"
    cfg_weight_decay="0.0"
    cfg_num_train_epochs="1"
    cfg_eval_steps="50"

    # Dataset
    cfg_train_dataset=""
    cfg_target_task=""
    cfg_percentage=""

    # Extras
    cfg_record_selections="false"
    cfg_record_selections_freq="1"
    cfg_optimizer_aware_diagnostic_interval="0"
    cfg_track_selection_domains="true"
    cfg_track_selection_domains_freq="10"
    cfg_val_seq_length_multiplier="1.2"
    cfg_update_compressor_freq="200"

    # Optional directory holding shared method YAMLs (see
    # configs/dolci32k_methods). Resolved relative to SFT/train/ unless
    # absolute. A method YAML present in the setting's own dir always wins.
    cfg_method_config_dir=""

    # Logging
    cfg_report_to="${REPORT_TO:-none}"
    cfg_wandb_project="${WANDB_PROJECT:-drpt-opus-sft}"
    cfg_wandb_run_name="${WANDB_RUN_NAME:-}"
    cfg_wandb_group="${WANDB_GROUP:-}"
    cfg_wandb_tags="${WANDB_TAGS:-}"
}

parse_yaml() {
    local file="$1"
    local section=""
    while IFS= read -r line; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        # Strip trailing inline comments. Only a '#' preceded by whitespace
        # starts one, so values such as `normal-512*512` are untouched.
        line="${line%%[[:space:]]#*}"
        [[ -z "${line// }" ]] && continue

        local key="" val=""
        if [[ "$line" =~ ^[[:space:]] ]]; then
            val=$(echo "$line" | cut -d: -f2- | xargs | sed 's/^"//;s/"$//' | sed "s/^'//;s/'$//")
            key="${section}.$(echo "$line" | cut -d: -f1 | xargs)"
        else
            local top_key top_val
            top_key=$(echo "$line" | cut -d: -f1 | xargs)
            top_val=$(echo "$line" | cut -d: -f2- | xargs | sed 's/^"//;s/"$//' | sed "s/^'//;s/'$//")
            if [[ -z "$top_val" ]]; then
                section="$top_key"; continue
            fi
            section=""
            key="$top_key"
            val="$top_val"
        fi

        case "$key" in
            method)                              cfg_method="$val" ;;
            finetuning)                          cfg_finetuning="$val" ;;
            # New nested structure: scoring.method, scoring.compression
            scoring.method)                      cfg_scoring_method="$val" ;;
            scoring.compression)                 cfg_score_compression="$val" ;;
            # New nested structure: optimizer.compression, optimizer.refresh_freq
            optimizer.type)                      cfg_optimizer_type="$val" ;;
            optimizer.muon_learning_rate)        cfg_muon_learning_rate="$val" ;;
            optimizer.aux_adamw_learning_rate)   cfg_aux_adamw_learning_rate="$val" ;;
            optimizer.compression)               cfg_opt_compression="$val" ;;
            optimizer.refresh_freq)              cfg_update_compressor_freq="$val" ;;
            # Legacy keys (backward compatibility)
            scoring_method)                      cfg_scoring_method="$val" ;;
            muon_learning_rate)                  cfg_muon_learning_rate="$val" ;;
            aux_adamw_learning_rate)             cfg_aux_adamw_learning_rate="$val" ;;
            score_grad_compression.sparsifier)   cfg_score_compression="$val" ;;
            score_grad_compression.projector)    ;; # Ignored in new format
            opt_grad_compression.sparsifier)     cfg_opt_compression="$val" ;;
            opt_grad_compression.projector)      ;; # Ignored in new format
            # Curation
            selection_frac)                      cfg_selection_frac="$val" ;;
            selection_mode)                      cfg_selection_mode="$val" ;;
            n_val)                               cfg_n_val="$val" ;;
            val_batch_size)                      cfg_val_batch_size="$val" ;;
            val_strategy)                        cfg_val_strategy="$val" ;;
            use_second_order)                    cfg_use_second_order="$val" ;;
            subset_mode)                         cfg_subset_mode="$val" ;;
            optimizer_aware.matrix_geometry)     cfg_oa_matrix_geometry="$val" ;;
            optimizer_aware.vector_geometry)     cfg_oa_vector_geometry="$val" ;;
            optimizer_aware.target_mode|optimizer_aware.method|optimizer_aware.selection_space) cfg_oa_target_mode="$val" ;;
            optimizer_aware.adam_eps)            cfg_oa_adam_eps="$val" ;;
            optimizer_aware.muon_reference)      cfg_oa_muon_reference="$val" ;;
            optimizer_aware.muon_momentum)       cfg_oa_muon_momentum="$val" ;;
            optimizer_aware.muon_nesterov)       cfg_oa_muon_nesterov="$val" ;;
            optimizer_aware.muon_steps)          cfg_oa_muon_steps="$val" ;;
            optimizer_aware.muon_eps)            cfg_oa_muon_eps="$val" ;;
            optimizer_aware.muon_max_dim)        cfg_oa_muon_max_dim="$val" ;;
            optimizer_aware.muon_lr_shape_scale) cfg_oa_muon_lr_shape_scale="$val" ;;
            optimizer_aware.muon_adjust_lr_fn)   cfg_oa_muon_adjust_lr_fn="$val" ;;
            optimizer_aware.muon_backend)        cfg_oa_muon_backend="$val" ;;
            optimizer_aware.lora_optimizer)      cfg_oa_lora_optimizer="$val" ;;
            optimizer_aware.spectral_lambda)     cfg_oa_spectral_lambda="$val" ;;
            optimizer_aware.spectral_eps)        cfg_oa_spectral_eps="$val" ;;
            optimizer_aware.token_normalized_selection) cfg_oa_token_normalized_selection="$val" ;;
            optimizer_aware.diagnostic_interval|optimizer_aware_diagnostic_interval) cfg_optimizer_aware_diagnostic_interval="$val" ;;
            soft_weighting.steps)                cfg_soft_steps="$val" ;;
            soft_weighting.lr)                   cfg_soft_lr="$val" ;;
            soft_weighting.tol)                  cfg_soft_tol="$val" ;;
            soft_weighting.patience)             cfg_soft_patience="$val" ;;
            soft_weighting.gamma)                cfg_soft_gamma="$val" ;;
            soft_weighting.use_optimizer_state)  cfg_soft_use_optimizer_state="$val" ;;
            soft_weighting.constraint)           cfg_soft_constraint="$val" ;;
            soft_weighting.replay_precision)     cfg_soft_replay_precision="$val" ;;
            muon_surrogate.alpha)                cfg_muon_sur_alpha="$val" ;;
            muon_surrogate.include_adamw_scores) cfg_muon_sur_include_adamw_scores="$val" ;;
            muon_surrogate.mode_weighting)       cfg_muon_sur_mode_weighting="$val" ;;
            muon_surrogate.saturation)           cfg_muon_sur_saturation="$val" ;;
            muon_surrogate.rank)                 cfg_muon_sur_rank="$val" ;;
            muon_surrogate.full_svd_max_dim)     cfg_muon_sur_full_svd_max_dim="$val" ;;
            muon_surrogate.rtol)                 cfg_muon_sur_rtol="$val" ;;
            muon_surrogate.oversample)           cfg_muon_sur_oversample="$val" ;;
            muon_surrogate.power_iters)          cfg_muon_sur_power_iters="$val" ;;
            lora_r)                              cfg_lora_r="$val" ;;
            lora_alpha)                          cfg_lora_alpha="$val" ;;
            lora_dropout)                        cfg_lora_dropout="$val" ;;
            experiment_profile)                 cfg_experiment_profile="$val" ;;
            setting_id)                         cfg_setting_id="$val" ;;
            artifact_build_id)                  cfg_artifact_build_id="$val" ;;
            model_profile)                      cfg_model_profile="$val" ;;
            model)                               cfg_model="$val" ;;
            model_revision)                      cfg_model_revision="$val" ;;
            tokenizer)                           cfg_tokenizer="$val" ;;
            tokenizer_revision)                  cfg_tokenizer_revision="$val" ;;
            seed)                                cfg_seed="$val" ;;
            batch_size)                          cfg_batch_size="$val" ;;
            gradient_accumulation_steps)         cfg_gradient_accumulation_steps="$val" ;;
            n_eval)                              cfg_n_eval="$val" ;;
            n_target_val)                       cfg_n_target_val="$val" ;;
            optim)                               cfg_optim="$val" ;;
            use_flash_attention)                 cfg_use_flash_attention="$val" ;;
            gradient_checkpointing)              cfg_gradient_checkpointing="$val" ;;
            tf32)                                cfg_tf32="$val" ;;
            dataloader_drop_last)                cfg_dataloader_drop_last="$val" ;;
            learning_rate)                       cfg_learning_rate="$val" ;;
            logical_candidate_batch_size)        cfg_logical_candidate_batch_size="$val" ;;
            candidate_microbatch_size)           cfg_candidate_microbatch_size="$val" ;;
            target_microbatch_size)              cfg_target_microbatch_size="$val" ;;
            target_signal_mode|target_signal)    cfg_target_signal_mode="$val" ;;
            target_signal_beta)                  cfg_target_signal_beta="$val" ;;
            target_signal_margin)                cfg_target_signal_margin="$val" ;;
            target_signal_incorrect_reward)      cfg_target_signal_incorrect_reward="$val" ;;
            target_signal_max_candidates)        cfg_target_signal_max_candidates="$val" ;;
            target_signal_groups_per_microbatch) cfg_target_signal_groups_per_microbatch="$val" ;;
            target_signal_align_prompts)         cfg_target_signal_align_prompts="$val" ;;
            target_cache_device)                 cfg_target_cache_device="$val" ;;
            target_cache_pin_memory)             cfg_target_cache_pin_memory="$val" ;;
            max_seq_length)                      cfg_max_seq_length="$val" ;;
            lr_scheduler_type)                   cfg_lr_scheduler_type="$val" ;;
            warmup_ratio)                        cfg_warmup_ratio="$val" ;;
            weight_decay)                        cfg_weight_decay="$val" ;;
            num_train_epochs)                    cfg_num_train_epochs="$val" ;;
            eval_steps)                          cfg_eval_steps="$val" ;;
            train_dataset)                       cfg_train_dataset="$val" ;;
            target_task)                         cfg_target_task="$val" ;;
            percentage)                          cfg_percentage="$val" ;;
            record_selections)                   cfg_record_selections="$val" ;;
            record_selections_freq)              cfg_record_selections_freq="$val" ;;
            track_selection_domains)             cfg_track_selection_domains="$val" ;;
            track_selection_domains_freq)        cfg_track_selection_domains_freq="$val" ;;
            val_seq_length_multiplier)           cfg_val_seq_length_multiplier="$val" ;;
            method_config_dir)                   cfg_method_config_dir="$val" ;;
            update_compressor_freq)              cfg_update_compressor_freq="$val" ;;
            report_to)                           cfg_report_to="$val" ;;
            wandb.project|wandb_project)          cfg_wandb_project="$val" ;;
            wandb.run_name|wandb.name|wandb_run_name) cfg_wandb_run_name="$val" ;;
            wandb.group|wandb_group)              cfg_wandb_group="$val" ;;
            wandb.tags|wandb_tags)                cfg_wandb_tags="$val" ;;
        esac
    done < "$file"
}

method_alias_to_exp() {
    case "$1" in
        FullTraining|Full-Training) echo "FullTraining-Full" ;;
        GlobalRaw|Global-Raw) echo "GlobalSubset-Full" ;;
        LayerwiseRaw|LayerWiseRaw|Layerwise-Raw|LayerWise-Raw) echo "LayerWiseSubset-Full" ;;
        OptGroupRaw|OptGroup-Raw) echo "OptimizerGroupWise-Full" ;;
        GlobalOptA|Global-OptA) echo "OptimizerAwareGlobalSubset-OptA-Full" ;;
        LayerwiseOptA|LayerWiseOptA|Layerwise-OptA|LayerWise-OptA) echo "LayerWiseOptimizerAwareSubset-OptA-Full" ;;
        GlobalOptANorm|Global-OptANorm) echo "OptimizerAwareGlobalSubset-OptANorm-Full" ;;
        LayerwiseOptANorm|LayerWiseOptANorm|Layerwise-OptANorm|LayerWise-OptANorm) echo "LayerWiseOptimizerAwareSubset-OptANorm-Full" ;;
        OptGroupOptA|OptGroup-OptA) echo "OptimizerAwareGroupWise-OptA-Full" ;;
        GlobalOptB|Global-OptB) echo "OptimizerAwareGlobalSubset-OptB-Full" ;;
        LayerwiseOptB|LayerWiseOptB|Layerwise-OptB|LayerWise-OptB) echo "LayerWiseOptimizerAwareSubset-OptB-Full" ;;
        OptGroupOptB|OptGroup-OptB) echo "OptimizerAwareGroupWise-OptB-Full" ;;
        GlobalRandomSubset|GlobalRandomSubset-Full|GlobalRandom) echo "GlobalRandomSubset-Full" ;;
        LayerWiseRandomSubset|LayerWiseRandomSubset-Full|LayerwiseRandom|LayerWiseRandom) echo "LayerWiseRandomSubset-Full" ;;
        GlobalSoftWeighting|GlobalSoftWeighting-Full|GlobalSoft) echo "GlobalSoftWeighting-Full" ;;
        LayerWiseSoftWeighting|LayerWiseSoftWeighting-Full|LayerwiseSoft|LayerWiseSoft) echo "LayerWiseSoftWeighting-Full" ;;
        LayerWiseSoftProbability|LayerWiseSoftProbability-Full|LayerwiseSoftP|LayerWiseSoftP) echo "LayerWiseSoftProbability-Full" ;;
        GlobalMuonSpectral|GlobalMuonSpectral-Full|GlobalHybridMuonSur) echo "GlobalMuonSpectral-Full" ;;
        LayerWiseMuonSpectral|LayerWiseMuonSpectral-Full|LayerwiseHybridMuonSur|LayerWiseHybridMuonSur) echo "LayerWiseMuonSpectral-Full" ;;
        GlobalMuonMatrixSpectral|GlobalMuonMatrixSpectral-Full|GlobalMuonSur|GlobalHybridMuonMatrixSur|GlobalMuonMatrixSur|GlobalMuonOnlySur) echo "GlobalMuonMatrixSpectral-Full" ;;
        LayerWiseMuonMatrixSpectral|LayerWiseMuonMatrixSpectral-Full|LayerwiseMuonSur|LayerWiseMuonSur|LayerwiseHybridMuonMatrixSur|LayerWiseHybridMuonMatrixSur|LayerwiseMuonMatrixSur|LayerWiseMuonMatrixSur|LayerwiseMuonOnlySur|LayerWiseMuonOnlySur) echo "LayerWiseMuonMatrixSpectral-Full" ;;
        LayerWiseMuonMatrixSpectralP|LayerWiseMuonMatrixSpectralP-Full|LayerwiseMuonPSur|LayerWiseMuonPSur|LayerwiseMuonMatrixPSur|LayerWiseMuonMatrixPSur|LayerwiseMuonOnlyPSur|LayerWiseMuonOnlyPSur) echo "LayerWiseMuonMatrixSpectralP-Full" ;;
        LayerWiseMuonMatrixSpectralSat|LayerWiseMuonMatrixSpectralSat-Full|LayerwiseMuonSatSur|LayerWiseMuonSatSur|LayerwiseMuonMatrixSatSur|LayerWiseMuonMatrixSatSur|LayerwiseMuonOnlySatSur|LayerWiseMuonOnlySatSur) echo "LayerWiseMuonMatrixSpectralSat-Full" ;;
        LayerWiseMuonMatrixSpectralSatP|LayerWiseMuonMatrixSpectralSatP-Full|LayerwiseMuonSatPSur|LayerWiseMuonSatPSur|LayerwiseMuonMatrixSatPSur|LayerWiseMuonMatrixSatPSur|LayerwiseMuonOnlySatPSur|LayerWiseMuonOnlySatPSur) echo "LayerWiseMuonMatrixSpectralSatP-Full" ;;
        *) echo "$1" ;;
    esac
}

method_display_label() {
    case "$1" in
        GlobalHybridMuonMatrixSur) echo "GlobalHybridMuonMatrixSur"; return ;;
        LayerwiseHybridMuonMatrixSur|LayerWiseHybridMuonMatrixSur)
            echo "LayerwiseHybridMuonMatrixSur"; return ;;
    esac
    case "$(method_alias_to_exp "$1")" in
        FullTraining-Full) echo "FullTraining" ;;
        GlobalSubset-Full) echo "GlobalRaw" ;;
        LayerWiseSubset-Full) echo "LayerwiseRaw" ;;
        OptimizerGroupWise-Full) echo "OptGroupRaw" ;;
        OptimizerAwareGlobalSubset-Full|OptimizerAwareGlobalSubset-OptA-Full) echo "GlobalOptA" ;;
        LayerWiseOptimizerAwareSubset-Full|LayerWiseOptimizerAwareSubset-OptA-Full) echo "LayerwiseOptA" ;;
        OptimizerAwareGlobalSubset-OptANorm-Full) echo "GlobalOptANorm" ;;
        LayerWiseOptimizerAwareSubset-OptANorm-Full) echo "LayerwiseOptANorm" ;;
        OptimizerAwareGroupWise-Full|OptimizerAwareGroupWise-OptA-Full) echo "OptGroupOptA" ;;
        OptimizerAwareGlobalSubset-OptB-Full) echo "GlobalOptB" ;;
        LayerWiseOptimizerAwareSubset-OptB-Full) echo "LayerwiseOptB" ;;
        OptimizerAwareGroupWise-OptB-Full) echo "OptGroupOptB" ;;
        GlobalRandomSubset-Full) echo "GlobalRandom" ;;
        LayerWiseRandomSubset-Full) echo "LayerwiseRandom" ;;
        GlobalSoftWeighting-Full) echo "GlobalSoft" ;;
        LayerWiseSoftWeighting-Full) echo "LayerwiseSoft" ;;
        LayerWiseSoftProbability-Full) echo "LayerwiseSoftP" ;;
        GlobalMuonSpectral-Full) echo "GlobalHybridMuonSur" ;;
        LayerWiseMuonSpectral-Full) echo "LayerwiseHybridMuonSur" ;;
        GlobalMuonMatrixSpectral-Full) echo "GlobalMuonSur" ;;
        LayerWiseMuonMatrixSpectral-Full) echo "LayerwiseMuonSur" ;;
        LayerWiseMuonMatrixSpectralP-Full) echo "LayerwiseMuonPSur" ;;
        LayerWiseMuonMatrixSpectralSat-Full) echo "LayerwiseMuonSatSur" ;;
        LayerWiseMuonMatrixSpectralSatP-Full) echo "LayerwiseMuonSatPSur" ;;
        *) echo "$1" ;;
    esac
}

optimizer_aware_virtual_base() {
    case "$1" in
        OptimizerAwareGlobalSubset-OptA-Full|OptimizerAwareGlobalSubset-OptANorm-Full|OptimizerAwareGlobalSubset-OptB-Full)
            echo "OptimizerAwareGlobalSubset-Full" ;;
        LayerWiseOptimizerAwareSubset-OptA-Full|LayerWiseOptimizerAwareSubset-OptANorm-Full|LayerWiseOptimizerAwareSubset-OptB-Full)
            echo "LayerWiseOptimizerAwareSubset-Full" ;;
        OptimizerAwareGroupWise-OptA-Full|OptimizerAwareGroupWise-OptB-Full)
            echo "OptimizerAwareGroupWise-Full" ;;
        *)
            return 1 ;;
    esac
}

optimizer_aware_virtual_target_mode() {
    case "$1" in
        *-OptANorm-*) echo "opta" ;;
        *-OptA-*) echo "opta" ;;
        *-OptB-*) echo "optb" ;;
        *) return 1 ;;
    esac
}

method_config_exists() {
    local method_name
    method_name="$(method_alias_to_exp "$1")"
    local base=""
    if method_config_file "$method_name" >/dev/null; then
        return 0
    fi
    if base=$(optimizer_aware_virtual_base "$method_name"); then
        method_config_file "$base" >/dev/null
        return $?
    fi
    return 1
}

# =============================================================================
# Method resolution (categories auto-discover from config dir)
# =============================================================================
resolve_methods() {
    local input="$1"

    # Discover available methods from the setting dir plus the shared method dir
    local available=()
    local seen_methods=""
    for f in "$config_dir"/*.yaml "${shared_method_dir:+$shared_method_dir/}"*.yaml; do
        [[ ! -f "$f" ]] && continue
        local name=$(basename "$f" .yaml)
        [[ "$name" == "defaults" ]] && continue
        [[ ",$seen_methods," == *",$name,"* ]] && continue
        seen_methods="${seen_methods:+$seen_methods,}$name"
        available+=("$name")
    done

    local resolved=""
    IFS=',' read -ra items <<< "$input"
    for item in "${items[@]}"; do
        item=$(echo "$item" | xargs)
        case "$item" in
            all)            for m in "${available[@]}"; do label=$(method_display_label "$m"); resolved="${resolved:+$resolved,}$label"; done ;;
            full-training)       for m in "${available[@]}"; do [[ "$m" == FullTraining-* ]] && label=$(method_display_label "$m") && resolved="${resolved:+$resolved,}$label"; done ;;
            layer-wise-subset)      for m in "${available[@]}"; do [[ "$m" == LayerWiseSubset-* ]] && label=$(method_display_label "$m") && resolved="${resolved:+$resolved,}$label"; done ;;
            optimizer-aware) for m in "${available[@]}"; do [[ "$m" == *OptimizerAware* ]] && label=$(method_display_label "$m") && resolved="${resolved:+$resolved,}$label"; done ;;
            optimizer-aware-groupwise) for m in "${available[@]}"; do [[ "$m" == OptimizerAwareGroupWise-* ]] && label=$(method_display_label "$m") && resolved="${resolved:+$resolved,}$label"; done ;;
            optimizer-group-wise|optgroup) for m in "${available[@]}"; do [[ "$m" == OptimizerGroupWise-* || "$m" == OptimizerAwareGroupWise-* ]] && label=$(method_display_label "$m") && resolved="${resolved:+$resolved,}$label"; done ;;
            sft10|optimizer-ablation-10|drpt-opus-10)
                for m in "${SFT10_METHODS[@]}"; do
                    method_config_exists "$m" && resolved="${resolved:+$resolved,}$m"
                done ;;
            new-solvers)
                for m in "${NEW_SOLVER_METHODS[@]}"; do
                    method_config_exists "$m" && resolved="${resolved:+$resolved,}$m"
                done ;;
            adamw-core|focused-adamw)
                for m in "${ADAMW_CORE_METHODS[@]}"; do
                    method_config_exists "$m" && resolved="${resolved:+$resolved,}$m"
                done ;;
            muon-core|focused-muon)
                for m in "${MUON_CORE_METHODS[@]}"; do
                    method_config_exists "$m" && resolved="${resolved:+$resolved,}$m"
                done ;;
            baseline9-adamw)
                for m in "${ADAMW_CORE_METHODS[@]}"; do
                    method_config_exists "$m" && resolved="${resolved:+$resolved,}$m"
                done ;;
            baseline9-muon)
                for m in "${MUON_CORE_METHODS[@]}"; do
                    method_config_exists "$m" && resolved="${resolved:+$resolved,}$m"
                done ;;
            hybrid-core|focused-hybrid)
                for m in "${HYBRID_CORE_METHODS[@]}"; do
                    method_config_exists "$m" && resolved="${resolved:+$resolved,}$m"
                done ;;
            soft-variants|soft-weighting-variants)
                for m in "${SOFT_VARIANT_METHODS[@]}"; do
                    method_config_exists "$m" && resolved="${resolved:+$resolved,}$m"
                done ;;
            muon-surrogate-variants|muon-surrogates)
                for m in "${MUON_SURROGATE_VARIANT_METHODS[@]}"; do
                    method_config_exists "$m" && resolved="${resolved:+$resolved,}$m"
                done ;;
            opus-baselines|drpt-opus-baselines)
                local baseline_order=(
                    FullTraining-Full
                    GlobalSubset-Full
                    OptimizerAwareGlobalSubset-Full
                    LayerWiseSubset-Full
                    LayerWiseOptimizerAwareSubset-Full
                    OptimizerGroupWise-Full
                    OptimizerAwareGroupWise-Full
                )
                for m in "${baseline_order[@]}"; do
                    [[ -f "$config_dir/${m}.yaml" ]] && label=$(method_display_label "$m") && resolved="${resolved:+$resolved,}$label"
                done ;;
            global-subset)         for m in "${available[@]}"; do [[ "$m" == GlobalSubset-* ]] && label=$(method_display_label "$m") && resolved="${resolved:+$resolved,}$label"; done ;;
            full)           for m in "${available[@]}"; do [[ "$m" == *-Full ]] && label=$(method_display_label "$m") && resolved="${resolved:+$resolved,}$label"; done ;;
            lora)           for m in "${available[@]}"; do [[ "$m" == *-LoRA ]] && label=$(method_display_label "$m") && resolved="${resolved:+$resolved,}$label"; done ;;
            meso)           for m in "${available[@]}"; do [[ "$m" == *-MeSO ]] && label=$(method_display_label "$m") && resolved="${resolved:+$resolved,}$label"; done ;;
            *)
                if method_config_exists "$item"; then
                    label=$(method_display_label "$item")
                    resolved="${resolved:+$resolved,}$label"
                else
                    echo "ERROR: Unknown method or category: $item" >&2
                    echo "Available: ${available[*]}" >&2
                    return 1
                fi ;;
        esac
    done

    echo "$resolved" | tr ',' '\n' | awk '!seen[$0]++' | tr '\n' ',' | sed 's/,$//'
}

# =============================================================================
# Run a single method
# =============================================================================
run_method() {
    local requested_name="$1"
    local exp_name
    exp_name="$(method_alias_to_exp "$requested_name")"
    local method_label
    method_label="$(method_display_label "$requested_name")"
    local config_stem="$exp_name"
    local virtual_target_mode=""
    local virtual_token_normalized_selection=""
    local virtual_base=""
    if virtual_base=$(optimizer_aware_virtual_base "$exp_name"); then
        config_stem="$virtual_base"
        virtual_target_mode=$(optimizer_aware_virtual_target_mode "$exp_name")
        case "$exp_name" in
            *-OptANorm-*) virtual_token_normalized_selection="true" ;;
            *) virtual_token_normalized_selection="false" ;;
        esac
    fi
    local config_file
    if ! config_file="$(method_config_file "$config_stem")"; then
        echo "ERROR: Config not found: ${config_stem}.yaml in $config_dir${shared_method_dir:+ or $shared_method_dir}"
        return 1
    fi

    # Load config: reset → setting defaults → model overlay → method.
    reset_config
    [[ -f "$config_dir/defaults.yaml" ]] && parse_yaml "$config_dir/defaults.yaml"
    [[ -n "$model_profile_override" ]] && cfg_model_profile="$model_profile_override"
    if [[ "$cfg_experiment_profile" == "dolci32k" ]]; then
        local model_overlay="$SCRIPT_DIR/configs/dolci32k/models/${cfg_model_profile}.yaml"
        if [[ ! -f "$model_overlay" ]]; then
            echo "ERROR: unknown dolci32k model profile: ${cfg_model_profile:-unset}" >&2
            return 1
        fi
        parse_yaml "$model_overlay"
    fi
    parse_yaml "$config_file"
    [[ -n "$virtual_target_mode" ]] && cfg_oa_target_mode="$virtual_target_mode"
    [[ -n "$virtual_token_normalized_selection" ]] && \
        cfg_oa_token_normalized_selection="$virtual_token_normalized_selection"

    # CLI overrides
    [[ -n "$target_signal_override" ]] && cfg_target_signal_mode="$target_signal_override"
    [[ -n "$target_signal_beta_override" ]] && cfg_target_signal_beta="$target_signal_beta_override"
    [[ -n "$target_signal_margin_override" ]] && cfg_target_signal_margin="$target_signal_margin_override"
    [[ -n "$target_signal_incorrect_reward_override" ]] && \
        cfg_target_signal_incorrect_reward="$target_signal_incorrect_reward_override"
    [[ -n "$target_signal_align_prompts_override" ]] && \
        cfg_target_signal_align_prompts="$target_signal_align_prompts_override"
    [[ -n "$seed_override" ]] && cfg_seed="$seed_override"
    [[ -n "$lr_override" ]] && cfg_learning_rate="$lr_override"
    [[ -n "$muon_lr_override" ]] && cfg_muon_learning_rate="$muon_lr_override"
    [[ -n "$aux_adamw_lr_override" ]] && cfg_aux_adamw_learning_rate="$aux_adamw_lr_override"
    [[ -n "$optimizer_type_override" ]] && cfg_optimizer_type="$optimizer_type_override"
    [[ -n "$report_to_override" ]] && cfg_report_to="$report_to_override"
    [[ -n "$wandb_project_override" ]] && cfg_wandb_project="$wandb_project_override"
    [[ -n "$wandb_run_name_override" ]] && cfg_wandb_run_name="$wandb_run_name_override"
    [[ -n "$wandb_group_override" ]] && cfg_wandb_group="$wandb_group_override"
    [[ -n "$wandb_tags_override" ]] && cfg_wandb_tags="$wandb_tags_override"
    [[ -n "$artifact_build_id_override" ]] && cfg_artifact_build_id="$artifact_build_id_override"
    if [[ -n "$candidate_microbatch_override" ]]; then
        if [[ "$cfg_experiment_profile" != "dolci32k" ]]; then
            echo "ERROR: --candidate-microbatch-size is supported only by dolci32k" >&2
            return 1
        fi
        cfg_candidate_microbatch_size="$candidate_microbatch_override"
    fi

    # dolci32k is a versioned experimental contract, not a loose collection of
    # defaults. Fail before model allocation if a method/config/CLI override
    # changes the logical selection window or one of the fixed campaign knobs.
    if [[ "$cfg_experiment_profile" == "dolci32k" ]]; then
        local profile_contract expected_train expected_target expected_model
        local expected_revision expected_tokenizer expected_tokenizer_revision
        local expected_max_seq_length
        if ! profile_contract="$(
            PYTHONDONTWRITEBYTECODE=1 "$DRPT_PYTHON" -c '
import sys
from SFT.data.dolci32k.profile import MAX_SEQ_LEN, MODEL_PROFILES, SETTINGS
setting = SETTINGS.get(sys.argv[1])
model = MODEL_PROFILES.get(sys.argv[2])
if setting is None:
    raise SystemExit(f"unknown dolci32k setting: {sys.argv[1]}")
if model is None:
    raise SystemExit(f"unknown dolci32k model profile: {sys.argv[2]}")
print(setting["general_pool"], setting["target"],
      model["model_name_or_path"], model["model_revision"],
      model["tokenizer_name"], model["tokenizer_revision"], MAX_SEQ_LEN,
      sep="\t")
' "$cfg_setting_id" "$cfg_model_profile"
        )"; then
            echo "ERROR: invalid dolci32k setting/model registry contract" >&2
            return 1
        fi
        IFS=$'\t' read -r expected_train expected_target expected_model \
            expected_revision expected_tokenizer expected_tokenizer_revision \
            expected_max_seq_length \
            <<< "$profile_contract"
        local dolci_checks=(
            "model_profile|$cfg_model_profile|$cfg_model_profile"
            "model|$cfg_model|$expected_model"
            "model_revision|$cfg_model_revision|$expected_revision"
            "tokenizer|$cfg_tokenizer|$expected_tokenizer"
            "tokenizer_revision|$cfg_tokenizer_revision|$expected_tokenizer_revision"
            "train_dataset|$cfg_train_dataset|$expected_train"
            "target_task|$cfg_target_task|$expected_target"
            "finetuning|$cfg_finetuning|Full"
            "batch_size|$cfg_batch_size|16"
            "gradient_accumulation_steps|$cfg_gradient_accumulation_steps|1"
            "max_seq_length|$cfg_max_seq_length|$expected_max_seq_length"
            "lr_scheduler_type|$cfg_lr_scheduler_type|linear"
            "num_train_epochs|$cfg_num_train_epochs|1"
            "eval_steps|$cfg_eval_steps|100"
            "n_eval|$cfg_n_eval|512"
            "n_target_val|$cfg_n_target_val|128"
            "n_val|$cfg_n_val|64"
            "val_batch_size|$cfg_val_batch_size|2"
            "val_strategy|$cfg_val_strategy|separate_batch"
            "selection_frac|$cfg_selection_frac|0.5"
            "selection_mode|$cfg_selection_mode|topk"
            "subset_mode|$cfg_subset_mode|one_pass"
            "logical_candidate_batch_size|$cfg_logical_candidate_batch_size|16"
            "target_cache_pin_memory|$cfg_target_cache_pin_memory|false"
            "use_flash_attention|$cfg_use_flash_attention|true"
            "gradient_checkpointing|$cfg_gradient_checkpointing|true"
            "tf32|${cfg_tf32,,}|true"
            "dataloader_drop_last|$cfg_dataloader_drop_last|true"
        )
        local check name actual expected
        for check in "${dolci_checks[@]}"; do
            IFS='|' read -r name actual expected <<< "$check"
            if [[ "$actual" != "$expected" ]]; then
                echo "ERROR: dolci32k requires $name=$expected, got $actual" >&2
                return 1
            fi
        done
        # The two microbatch knobs only chunk GPU compute; they do not change
        # selection or optimizer semantics. tests/test_dolci32k_selection_window.py's
        # test_custom_autograd_gradient_and_single_outer_step_match_c16_c2_c1
        # pins candidate C=16/2/1 and target C=1/2 to identical selected indices,
        # parameter gradients, and post-step parameters. The logical window
        # itself (N=16 candidates, T=2 target) stays locked above, so only the
        # chunk sizes are free -- and only over the validated divisors of 16.
        case "$cfg_candidate_microbatch_size" in
            1|2|4|8) ;;
            16)
                if [[ "$allow_dolci_c16_probe" != "true" || -z "$max_steps_override" ]] || \
                   (( max_steps_override > 3 )); then
                    echo "ERROR: dolci32k C=16 is probe-only; pass --allow-dolci-c16-probe with explicit --max-steps <= 3" >&2
                    return 1
                fi
                export DRPT_DOLCI_C16_PROBE=1
                ;;
            *)
                echo "ERROR: dolci32k candidate_microbatch_size must be 1, 2, 4, or 8 (or guarded probe-only 16), got $cfg_candidate_microbatch_size" >&2
                return 1
                ;;
        esac
        case "$cfg_target_microbatch_size" in
            1|2) ;;
            *)
                echo "ERROR: dolci32k target_microbatch_size must be 1 or 2, got $cfg_target_microbatch_size" >&2
                return 1
                ;;
        esac
        case "$cfg_target_signal_mode" in
            nll|answer_only_ce) ;;
            reward_weighted_sft) ;;
            correct_incorrect_margin)
                # The margin loss compares two trajectories of one prompt, so
                # its chunk unit is whole prompt groups, not rows. Reward
                # weighting has no such coupling and keeps target_microbatch_size.
                if [[ -z "$cfg_target_signal_groups_per_microbatch" ]]; then
                    cfg_target_signal_groups_per_microbatch=1
                fi
                ;;
            *)
                echo "ERROR: target_signal_mode must be nll, answer_only_ce, correct_incorrect_margin, or reward_weighted_sft; got $cfg_target_signal_mode" >&2
                return 1
                ;;
        esac
        # Target-cache placement is pure storage location for the captured
        # fp32 target gradient; cuda avoids a model-sized host round trip per step.
        case "$cfg_target_cache_device" in
            cpu|cuda) ;;
            *)
                echo "ERROR: dolci32k target_cache_device must be cpu or cuda, got $cfg_target_cache_device" >&2
                return 1
                ;;
        esac
        for check in \
            "percentage|$cfg_percentage|1.0" \
            "learning_rate|$cfg_learning_rate|1e-5" \
            "warmup_ratio|$cfg_warmup_ratio|0.03" \
            "weight_decay|$cfg_weight_decay|0"; do
            IFS='|' read -r name actual expected <<< "$check"
            if ! awk -v actual="$actual" -v expected="$expected" \
                'BEGIN { exit !((actual + 0) == (expected + 0)) }'; then
                echo "ERROR: dolci32k requires $name=$expected, got $actual" >&2
                return 1
            fi
        done
        if [[ -n "$max_steps_override" ]] && (( max_steps_override > 2000 )); then
            echo "ERROR: dolci32k max_steps may only be a <=2000 smoke prefix" >&2
            return 1
        fi
        if [[ "$cfg_optimizer_type" != "adamw" && "$cfg_optimizer_type" != "muon" ]]; then
            echo "ERROR: dolci32k optimizer.type must be adamw or muon, got $cfg_optimizer_type" >&2
            return 1
        fi
        if [[ "$cfg_optimizer_type" == "muon" ]]; then
            if ! awk -v actual="$cfg_muon_learning_rate" 'BEGIN { exit !((actual + 0) == 3e-4) }'; then
                echo "ERROR: dolci32k Muon matrix LR must be 3e-4" >&2
                return 1
            fi
            if ! awk -v actual="$cfg_aux_adamw_learning_rate" 'BEGIN { exit !((actual + 0) == 1e-5) }'; then
                echo "ERROR: dolci32k auxiliary AdamW LR must be 1e-5" >&2
                return 1
            fi
        fi
    elif [[ "$retry_failed" == "true" ]]; then
        echo "ERROR: --retry-failed is supported only by immutable 32k profiles" >&2
        return 1
    fi

    # Canonical baseline labels encode scorer/optimizer semantics. Reject
    # accidental cross-family runs instead of silently producing a mislabeled
    # experiment directory.
    case "$method_label" in
        LayerwiseOptA)
            if [[ "$cfg_optimizer_type" != "adamw" ]]; then
                echo "ERROR: $method_label is AdamW-only; got optimizer.type=$cfg_optimizer_type"
                return 1
            fi
            ;;
        GlobalMuonSur|LayerwiseMuonSur|LayerwiseMuonPSur|LayerwiseMuonSatSur|LayerwiseMuonSatPSur)
            if [[ "$cfg_optimizer_type" != "muon" ]]; then
                echo "ERROR: $method_label is a Muon-only baseline; got optimizer.type=$cfg_optimizer_type"
                return 1
            fi
            ;;
        GlobalHybridMuonSur|LayerwiseHybridMuonSur|GlobalHybridMuonMatrixSur|LayerwiseHybridMuonMatrixSur)
            if [[ "$cfg_optimizer_type" != "hybrid" ]]; then
                echo "ERROR: $method_label is a legacy hybrid-score baseline; got optimizer.type=$cfg_optimizer_type"
                return 1
            fi
            ;;
    esac

    if [[ "$cfg_optimizer_type" == "muon" || "$cfg_optimizer_type" == "hybrid" ]]; then
        if [[ -n "$cfg_opt_compression" && "$cfg_opt_compression" != "none" ]]; then
            echo "ERROR: optimizer.type=$cfg_optimizer_type cannot use MeSO/update compression; refusing to replace the official-first Muon runtime"
            return 1
        fi
    fi

    # Validate required fields
    if [[ -z "$cfg_target_task" ]] || [[ -z "$cfg_percentage" ]]; then
        echo "ERROR: target_task and percentage must be set (in defaults.yaml or method config)"
        return 1
    fi

    case "$cfg_method" in
        GlobalRandomSubset|LayerWiseRandomSubset|GlobalSoftWeighting|LayerWiseSoftWeighting|LayerWiseSoftProbability|GlobalMuonSpectral|LayerWiseMuonSpectral|GlobalMuonMatrixSpectral|LayerWiseMuonMatrixSpectral|LayerWiseMuonMatrixSpectralP|LayerWiseMuonMatrixSpectralSat|LayerWiseMuonMatrixSpectralSatP)
            if [[ "$cfg_scoring_method" != "reduced_ghost" || "$cfg_selection_mode" != "topk" || "$cfg_subset_mode" != "one_pass" ]]; then
                echo "ERROR: $cfg_method v1 requires reduced_ghost, topk, and one_pass"
                return 1
            fi
            if [[ "$cfg_use_second_order" == "true" || -n "$cfg_score_compression" || -n "$cfg_opt_compression" ]]; then
                echo "ERROR: $cfg_method does not support second-order, score compression, or MeSO/update compression"
                return 1
            fi
            ;;
    esac
    if [[ "$cfg_soft_constraint" != "capped_simplex" && "$cfg_soft_constraint" != "probability_simplex" ]]; then
        echo "ERROR: soft_weighting.constraint must be capped_simplex or probability_simplex"
        return 1
    fi
    if [[ "$cfg_soft_replay_precision" != "fp32" && "$cfg_soft_replay_precision" != "bf16_fp32" ]]; then
        echo "ERROR: soft_weighting.replay_precision must be fp32 or bf16_fp32"
        return 1
    fi
    case "$cfg_method" in
        LayerWiseSoftProbability)
            if [[ "$cfg_soft_constraint" != "probability_simplex" ]]; then
                echo "ERROR: LayerWiseSoftProbability requires soft_weighting.constraint=probability_simplex"
                return 1
            fi
            if [[ "$cfg_optimizer_type" != "adamw" && "$cfg_optimizer_type" != "muon" ]]; then
                echo "ERROR: LayerWiseSoftProbability supports optimizer.type=adamw or muon only"
                return 1
            fi
            ;;
        GlobalSoftWeighting|LayerWiseSoftWeighting)
            if [[ "$cfg_soft_constraint" != "capped_simplex" ]]; then
                echo "ERROR: $cfg_method requires soft_weighting.constraint=capped_simplex"
                return 1
            fi
            ;;
        *)
            if [[ "$cfg_soft_constraint" != "capped_simplex" ]]; then
                echo "ERROR: probability_simplex is valid only for LayerWiseSoftProbability"
                return 1
            fi
            ;;
    esac
    case "$cfg_method" in
        GlobalMuonSpectral|LayerWiseMuonSpectral|GlobalMuonMatrixSpectral|LayerWiseMuonMatrixSpectral|LayerWiseMuonMatrixSpectralP|LayerWiseMuonMatrixSpectralSat|LayerWiseMuonMatrixSpectralSatP)
            if [[ "$cfg_optimizer_type" != "muon" && "$cfg_optimizer_type" != "hybrid" ]]; then
                echo "ERROR: $cfg_method requires optimizer.type=muon or hybrid"
                return 1
            fi
            if [[ "$cfg_optimizer_type" == "muon" ]]; then
                cfg_muon_sur_include_adamw_scores="false"
            fi
            if [[ "$cfg_oa_spectral_lambda" != "0" && "$cfg_oa_spectral_lambda" != "0.0" ]]; then
                echo "ERROR: $cfg_method does not use legacy optimizer_aware.spectral_lambda"
                return 1
            fi
            ;;
    esac
    if [[ "$cfg_muon_sur_include_adamw_scores" != "true" && "$cfg_muon_sur_include_adamw_scores" != "false" ]]; then
        echo "ERROR: muon_surrogate.include_adamw_scores must be true or false"
        return 1
    fi
    if [[ "$cfg_muon_sur_mode_weighting" != "uniform" && "$cfg_muon_sur_mode_weighting" != "singular_value" ]]; then
        echo "ERROR: muon_surrogate.mode_weighting must be uniform or singular_value"
        return 1
    fi
    if [[ "$cfg_muon_sur_saturation" != "true" && "$cfg_muon_sur_saturation" != "false" ]]; then
        echo "ERROR: muon_surrogate.saturation must be true or false"
        return 1
    fi
    case "$cfg_method" in
        GlobalMuonSpectral|GlobalMuonMatrixSpectral)
            if [[ "$cfg_muon_sur_saturation" == "true" ]]; then
                echo "ERROR: Muon surrogate saturation is currently layerwise-only"
                return 1
            fi
            ;;
    esac
    if [[ "$cfg_oa_token_normalized_selection" != "true" && "$cfg_oa_token_normalized_selection" != "false" ]]; then
        echo "ERROR: optimizer_aware.token_normalized_selection must be true or false"
        return 1
    fi
    if ! [[ "$cfg_optimizer_aware_diagnostic_interval" =~ ^[0-9]+$ ]]; then
        echo "ERROR: optimizer_aware.diagnostic_interval must be a non-negative integer"
        return 1
    fi
    case "$cfg_method" in
        GlobalMuonMatrixSpectral|LayerWiseMuonMatrixSpectral|LayerWiseMuonMatrixSpectralP|LayerWiseMuonMatrixSpectralSat|LayerWiseMuonMatrixSpectralSatP)
            if [[ "$cfg_muon_sur_include_adamw_scores" != "false" ]]; then
                echo "ERROR: $cfg_method requires muon_surrogate.include_adamw_scores=false"
                return 1
            fi
            ;;
    esac

    # Derived values
    local internal_method="NA"
    [[ "$cfg_method" != "FullTraining" ]] && internal_method="$cfg_method"

    local use_lora="false"
    [[ "$cfg_finetuning" == "LoRA" || "$cfg_finetuning" == "MeSO-LoRA" ]] && use_lora="true"

    # LR fallback if not specified anywhere (every YAML should set this explicitly)
    if [[ -z "$cfg_learning_rate" ]]; then
        [[ "$use_lora" == "true" ]] && cfg_learning_rate="1e-04" || cfg_learning_rate="1e-05"
    fi

    local model_name=$(basename "$cfg_model")
    local method_str="$method_label"
    [[ "$internal_method" != "NA" && "$cfg_use_second_order" == "true" ]] && method_str="${method_str}-2nd"
    local optimizer_label="$cfg_optimizer_type"

    # Build job name
    local train_str="${cfg_train_dataset:-default}"
    local JOB_NAME="${train_str}_${cfg_target_task}-${method_str}-${optimizer_label}-p${cfg_percentage}-lr${cfg_learning_rate}-b${cfg_batch_size}-v${cfg_n_val}-s${cfg_seed}-${model_name}"
    if [[ "$cfg_experiment_profile" == "dolci32k" ]]; then
        train_str="$cfg_setting_id"
        JOB_NAME="${cfg_setting_id}-${method_str}-${optimizer_label}-p${cfg_percentage}-lr${cfg_learning_rate}-b${cfg_batch_size}-v${cfg_n_val}-s${cfg_seed}-${model_name}"
    fi
    # A non-default target signal changes what the run optimizes, so it must not
    # land in the same output directory as the nll run it is compared against.
    # nll keeps its bare historical name so existing campaign paths still resolve.
    if [[ "$cfg_target_signal_mode" != "nll" ]]; then
        local signal_tag="${cfg_target_signal_mode//_/-}"
        JOB_NAME="${JOB_NAME}-ts${signal_tag}"
    fi
    # Prompt alignment shrinks D* to the margin-usable subset, so an aligned run
    # optimizes a different target set than an unaligned one of the same signal.
    # Without its own name the aligned nll control would overwrite the plain one.
    if [[ "$cfg_target_signal_align_prompts" == "true" ]]; then
        JOB_NAME="${JOB_NAME}-aligned"
    fi
    local default_wandb_run_name="$JOB_NAME"
    local default_wandb_group="${train_str}_${cfg_target_task}-${optimizer_label}-s${cfg_seed}"
    local default_wandb_tags="sft,${method_label},${cfg_method},${cfg_finetuning},${cfg_optimizer_type},${cfg_target_task},${cfg_val_strategy},${cfg_oa_target_mode}"
    case "$cfg_method" in
        GlobalSoftWeighting|LayerWiseSoftWeighting|LayerWiseSoftProbability)
            default_wandb_tags="${default_wandb_tags},soft-constraint-${cfg_soft_constraint},soft-replay-${cfg_soft_replay_precision}" ;;
    esac
    if [[ -n "$campaign_id_override" ]]; then
        default_wandb_run_name="${campaign_id_override}-${JOB_NAME}"
        default_wandb_group="${campaign_id_override}-${default_wandb_group}"
        default_wandb_tags="${default_wandb_tags},campaign-${campaign_id_override}"
    fi
    local wandb_run_name="${cfg_wandb_run_name:-$default_wandb_run_name}"
    local wandb_group="${cfg_wandb_group:-$default_wandb_group}"
    local wandb_tags="${cfg_wandb_tags:-$default_wandb_tags}"

    local data_dir="${DRPT_DATA_DIR:-$REPO_ROOT/SFT/data}"
    local runs_root="${DRPT_RUNS_DIR:-$REPO_ROOT/SFT/runs}"
    if [[ "$cfg_experiment_profile" == "dolci32k" && -z "$cfg_artifact_build_id" ]]; then
        local current_pointer="$dolci_artifact_root/CURRENT"
        if [[ "$dry_run" == "true" ]]; then
            cfg_artifact_build_id="CURRENT"
        elif [[ -f "$current_pointer" ]]; then
            cfg_artifact_build_id="$(<"$current_pointer")"
        else
            echo "ERROR: $cfg_experiment_profile CURRENT pointer is missing: $current_pointer" >&2
            return 1
        fi
    fi
    [[ -n "$campaign_id_override" ]] && runs_root="$runs_root/campaigns/$campaign_id_override"
    local output_dir="$runs_root/${JOB_NAME}"
    if [[ "$dry_run" != "true" ]]; then
        if [[ -e "$output_dir" ]]; then
            if [[ "$cfg_experiment_profile" == "dolci32k" ]]; then
                if [[ -f "$output_dir/_SUCCESS" ]] || \
                   { [[ -f "$output_dir/run_status.json" ]] && \
                     grep -Eq '"status"[[:space:]]*:[[:space:]]*"complete"' "$output_dir/run_status.json"; }; then
                    if [[ -f "$output_dir/run_status.json" ]] && \
                       grep -Eq '"artifact_build_id"[[:space:]]*:[[:space:]]*"'"$cfg_artifact_build_id"'"' "$output_dir/run_status.json"; then
                        echo "[protected] completed $cfg_experiment_profile run already exists: $output_dir"
                        return 0
                    fi
                    echo "ERROR: completed $cfg_experiment_profile run has missing/mismatched artifact provenance: $output_dir" >&2
                    return 1
                fi
                if [[ "$retry_failed" != "true" ]]; then
                    echo "ERROR: incomplete/failed $cfg_experiment_profile run already exists: $output_dir" >&2
                    echo "Re-run with --retry-failed to archive it and retry; completed runs can never be overwritten." >&2
                    return 1
                fi
                local retry_archive_root="$runs_root/.failed-retries"
                local retry_stamp
                retry_stamp="$(date -u +%Y%m%dT%H%M%SZ)-$$"
                mkdir -p "$retry_archive_root"
                local retry_archive="$retry_archive_root/${JOB_NAME}-${retry_stamp}"
                echo "[retry] archiving failed/incomplete run to $retry_archive"
                mv "$output_dir" "$retry_archive"
            elif [[ -n "$campaign_id_override" ]]; then
                echo "ERROR: campaign output already exists; refusing to overwrite: $output_dir"
                return 1
            fi
        fi
        mkdir -p "$output_dir"
    fi

    echo ""
    echo "=============================================="
    echo "  Running: $method_label"
    echo "=============================================="
    echo "Config: $config_file"
    echo "Job: $JOB_NAME"
    echo "Model: $cfg_model | Task: $cfg_target_task | LR: $cfg_learning_rate"
    echo "Method: $cfg_method | Label: $method_label | Finetuning: $cfg_finetuning"
    echo "Optimizer: $cfg_optimizer_type | OA target: $cfg_oa_target_mode | Logging: $cfg_report_to | wandb: $cfg_wandb_project/$wandb_run_name"
    if [[ "$cfg_optimizer_type" == "muon" || "$cfg_optimizer_type" == "hybrid" ]]; then
        echo "Optimizer LRs: Muon=${cfg_muon_learning_rate:-$cfg_learning_rate} | auxiliary AdamW=${cfg_aux_adamw_learning_rate:-$cfg_learning_rate}"
    fi
    echo "Batch: $cfg_batch_size | Val: $cfg_val_batch_size | Curation: $cfg_selection_frac"
    if [[ "$cfg_experiment_profile" == "dolci32k" ]]; then
        echo "Profile: $cfg_experiment_profile/$cfg_setting_id | artifact: $cfg_artifact_build_id"
        [[ -n "$cfg_model_profile" ]] && echo "Model profile: $cfg_model_profile"
        echo "Window: logical=$cfg_logical_candidate_batch_size candidate_micro=$cfg_candidate_microbatch_size target_logical=$cfg_val_batch_size target_micro=$cfg_target_microbatch_size target_cache=$cfg_target_cache_device"
        echo "Target signal: $cfg_target_signal_mode (beta=$cfg_target_signal_beta margin=$cfg_target_signal_margin incorrect_reward=$cfg_target_signal_incorrect_reward groups_per_micro=${cfg_target_signal_groups_per_microbatch:-all} align_prompts=$cfg_target_signal_align_prompts)"
    fi
    case "$cfg_method" in
        GlobalSoftWeighting|LayerWiseSoftWeighting|LayerWiseSoftProbability)
            echo "Soft constraint: $cfg_soft_constraint | replay precision: $cfg_soft_replay_precision" ;;
    esac
    [[ -n "$campaign_id_override" ]] && echo "Campaign: $campaign_id_override"
    [[ -n "$max_steps_override" ]] && echo "Max steps override: $max_steps_override"
    [[ "$cfg_candidate_microbatch_size" == "16" ]] && echo "C=16 guard: short diagnostic probe only"
    echo "Output: $output_dir"
    echo "=============================================="

    # FSDP for large models
    local fsdp_args=""
    case "$cfg_model" in
        *Llama-2-13b*|*llama-2-13b*)
            fsdp_args="--fsdp 'full_shard auto_wrap' --fsdp_config llama2_13b_finetune" ;;
        *Mistral-7B*|*mistral-7b*)
            fsdp_args="--fsdp 'full_shard auto_wrap' --fsdp_config mistral_7b_finetune" ;;
    esac

    local DATA_SEED=$((cfg_seed + 1))
    # Derive port from SLURM_JOB_ID (or PID fallback) so different jobs landing
    # on the same node get distinct ports. Random ports caused C10D rendezvous
    # collisions when slurm packed multiple jobs per node.
    local PORT=$((20000 + (${SLURM_JOB_ID:-$$} % 40000)))

    # Build command
    local cmd="torchrun --nproc_per_node 1 --nnodes 1 \
--rdzv_id=$RANDOM --rdzv_backend c10d --rdzv_endpoint=localhost:$PORT \
-m SFT.train.train \
$FIXED_ARGS \
$fsdp_args \
--report_to $cfg_report_to \
--run_name $wandb_run_name \
--wandb_project $cfg_wandb_project \
--wandb_run_name $wandb_run_name \
--wandb_group $wandb_group \
--wandb_tags $wandb_tags \
--max_seq_length $cfg_max_seq_length \
--lr_scheduler_type $cfg_lr_scheduler_type \
--warmup_ratio $cfg_warmup_ratio \
--weight_decay $cfg_weight_decay \
--num_train_epochs $cfg_num_train_epochs \
--eval_steps $cfg_eval_steps \
--model_name_or_path $cfg_model \
--model_revision $cfg_model_revision \
--output_dir $output_dir \
--data_dir $data_dir \
--percentage $cfg_percentage \
--data_seed $DATA_SEED \
--per_device_train_batch_size $cfg_batch_size \
--method $internal_method \
--n_val $cfg_n_val \
--n_eval $cfg_n_eval \
--analysis_dataset $cfg_target_task \
--learning_rate $cfg_learning_rate \
--gradient_accumulation_steps $cfg_gradient_accumulation_steps \
--gradient_checkpointing $cfg_gradient_checkpointing \
--dataloader_drop_last $cfg_dataloader_drop_last \
--tf32 $cfg_tf32 \
--seed $cfg_seed \
--optim $cfg_optim \
--optimizer_type $cfg_optimizer_type \
--selection_frac $cfg_selection_frac \
--selection_mode $cfg_selection_mode \
--val_strategy $cfg_val_strategy \
--scoring_method $cfg_scoring_method \
--subset_mode $cfg_subset_mode \
--optimizer_aware_matrix_geometry $cfg_oa_matrix_geometry \
--optimizer_aware_vector_geometry $cfg_oa_vector_geometry \
--optimizer_aware_target_mode $cfg_oa_target_mode \
--optimizer_aware_adam_eps $cfg_oa_adam_eps \
--optimizer_aware_muon_reference $cfg_oa_muon_reference \
--optimizer_aware_muon_momentum $cfg_oa_muon_momentum \
--optimizer_aware_muon_nesterov $cfg_oa_muon_nesterov \
--optimizer_aware_muon_steps $cfg_oa_muon_steps \
--optimizer_aware_muon_eps $cfg_oa_muon_eps \
--optimizer_aware_muon_max_dim $cfg_oa_muon_max_dim \
--optimizer_aware_muon_lr_shape_scale $cfg_oa_muon_lr_shape_scale \
--optimizer_aware_muon_adjust_lr_fn $cfg_oa_muon_adjust_lr_fn \
--optimizer_aware_muon_backend $cfg_oa_muon_backend \
--optimizer_aware_lora_optimizer $cfg_oa_lora_optimizer \
--optimizer_aware_spectral_lambda $cfg_oa_spectral_lambda \
--optimizer_aware_spectral_eps $cfg_oa_spectral_eps \
--optimizer_aware_token_normalized_selection $cfg_oa_token_normalized_selection \
--optimizer_aware_diagnostic_interval $cfg_optimizer_aware_diagnostic_interval \
--soft_weighting_steps $cfg_soft_steps \
--soft_weighting_lr $cfg_soft_lr \
--soft_weighting_tol $cfg_soft_tol \
--soft_weighting_patience $cfg_soft_patience \
--soft_weighting_gamma $cfg_soft_gamma \
--soft_weighting_use_optimizer_state $cfg_soft_use_optimizer_state \
--soft_weighting_constraint $cfg_soft_constraint \
--soft_replay_precision $cfg_soft_replay_precision \
--muon_surrogate_alpha $cfg_muon_sur_alpha \
--muon_surrogate_include_adamw_scores $cfg_muon_sur_include_adamw_scores \
--muon_surrogate_mode_weighting $cfg_muon_sur_mode_weighting \
--muon_surrogate_saturation $cfg_muon_sur_saturation \
--muon_surrogate_rank $cfg_muon_sur_rank \
--muon_surrogate_full_svd_max_dim $cfg_muon_sur_full_svd_max_dim \
--muon_surrogate_rtol $cfg_muon_sur_rtol \
--muon_surrogate_oversample $cfg_muon_sur_oversample \
--muon_surrogate_power_iters $cfg_muon_sur_power_iters \
--val_seq_length_multiplier $cfg_val_seq_length_multiplier \
--track_selection_domains $cfg_track_selection_domains \
--track_selection_domains_freq $cfg_track_selection_domains_freq \
--use_flash_attention $cfg_use_flash_attention"

    [[ -n "$cfg_tokenizer" ]] && cmd="$cmd --tokenizer_name $cfg_tokenizer"
    [[ -n "$cfg_tokenizer_revision" ]] && cmd="$cmd --tokenizer_revision $cfg_tokenizer_revision"

    # Optional args
    [[ -n "$max_steps_override" ]] && cmd="$cmd --max_steps $max_steps_override"
    if [[ "$cfg_optimizer_type" == "muon" || "$cfg_optimizer_type" == "hybrid" ]]; then
        [[ -n "$cfg_muon_learning_rate" ]] && cmd="$cmd --muon_learning_rate $cfg_muon_learning_rate"
        [[ -n "$cfg_aux_adamw_learning_rate" ]] && cmd="$cmd --aux_adamw_learning_rate $cfg_aux_adamw_learning_rate"
    fi
    [[ -n "$cfg_train_dataset" ]] && cmd="$cmd --train_dataset_names $cfg_train_dataset"
    [[ -n "$cfg_val_batch_size" ]] && cmd="$cmd --val_batch_size_for_selection $cfg_val_batch_size"
    if [[ "$cfg_experiment_profile" == "dolci32k" ]]; then
        cmd="$cmd --experiment_profile $cfg_experiment_profile \
--setting_id $cfg_setting_id \
--artifact_build_id $cfg_artifact_build_id \
--n_target_val $cfg_n_target_val \
--logical_candidate_batch_size $cfg_logical_candidate_batch_size \
--candidate_microbatch_size $cfg_candidate_microbatch_size \
--target_microbatch_size $cfg_target_microbatch_size \
--target_cache_device $cfg_target_cache_device \
--target_signal_mode $cfg_target_signal_mode \
--target_signal_beta $cfg_target_signal_beta \
--target_signal_margin $cfg_target_signal_margin \
--target_signal_incorrect_reward $cfg_target_signal_incorrect_reward \
--target_signal_align_prompts $cfg_target_signal_align_prompts \
--target_cache_pin_memory $cfg_target_cache_pin_memory"
        [[ -n "$cfg_target_signal_max_candidates" ]] && \
            cmd="$cmd --target_signal_max_candidates $cfg_target_signal_max_candidates"
        [[ -n "$cfg_target_signal_groups_per_microbatch" ]] && \
            cmd="$cmd --target_signal_groups_per_microbatch $cfg_target_signal_groups_per_microbatch"
        [[ -n "$cfg_model_profile" ]] && cmd="$cmd --model_profile $cfg_model_profile"
    fi

    # LoRA
    if [[ "$use_lora" == "true" ]]; then
        cmd="$cmd --lora True --lora_r $cfg_lora_r --lora_alpha $cfg_lora_alpha --lora_dropout $cfg_lora_dropout"
    else
        cmd="$cmd --lora False"
    fi

    # Scoring compression: parse "SPARSIFIER" or "SPARSIFIER/PROJECTOR" format
    if [[ -n "$cfg_score_compression" && "$cfg_score_compression" != "none" ]]; then
        cmd="$cmd --score_compression ${cfg_score_compression%%/*}"
    fi

    # Optimizer compression: parse "SPARSIFIER" or "SPARSIFIER/PROJECTOR" format
    if [[ -n "$cfg_opt_compression" && "$cfg_opt_compression" != "none" ]]; then
        local opt_sparsifier="${cfg_opt_compression%%/*}"
        cmd="$cmd --sparsification $opt_sparsifier --update_compressor_freq $cfg_update_compressor_freq"
        if [[ "$cfg_opt_compression" == */* ]]; then
            local opt_projector="${cfg_opt_compression##*/}"
            [[ "$opt_projector" != "none" ]] && cmd="$cmd --projection $opt_projector"
        fi
    fi

    # Second-order
    [[ "$internal_method" != "NA" && "$cfg_use_second_order" == "true" ]] && \
        cmd="$cmd --use_second_order True"

    # Recording
    [[ "$cfg_record_selections" == "true" ]] && \
        cmd="$cmd --record_selections True --record_selections_freq $cfg_record_selections_freq"

    # Eval split override
    [[ -n "$eval_split_override" ]] && cmd="$cmd --eval_split $eval_split_override"

    if [[ "$dry_run" == "true" ]]; then
        echo "[DRY-RUN] $cmd"
    else
        write_run_status() {
            local status="$1" exit_code="$2"
            local status_tmp="$output_dir/.run_status.json.tmp.$$"
            printf '{"schema_version":1,"profile":"%s","setting":"%s","method":"%s","optimizer_family":"%s","artifact_build_id":"%s","status":"%s","exit_code":%s,"updated_at":"%s"}\n' \
                "$cfg_experiment_profile" "$cfg_setting_id" "$method_label" \
                "$cfg_optimizer_type" "$cfg_artifact_build_id" "$status" \
                "$exit_code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status_tmp"
            mv "$status_tmp" "$output_dir/run_status.json"
        }
        if [[ "$cfg_experiment_profile" == "dolci32k" ]]; then
            export WANDB_DIR="$output_dir/wandb"
            mkdir -p "$WANDB_DIR"
            write_run_status running 0
        fi
        eval "$cmd" 2>&1 | tee "$output_dir/train.log"
        local train_exit=${PIPESTATUS[0]}
        if [[ "$cfg_experiment_profile" == "dolci32k" ]]; then
            if (( train_exit == 0 )); then
                write_run_status complete 0
                : > "$output_dir/_SUCCESS"
            else
                write_run_status failed "$train_exit"
            fi
        fi
        return "$train_exit"
    fi
}

# =============================================================================
# Main
# =============================================================================
if ! resolved_methods=$(resolve_methods "$methods"); then
    exit 1
fi
IFS=',' read -ra method_list <<< "$resolved_methods"
TOTAL=${#method_list[@]}

echo ""
echo "========================================================"
echo "  SFT Training"
echo "========================================================"
echo "Config dir: $config_dir"
echo "Methods: $resolved_methods ($TOTAL total)"
echo "========================================================"

current=0
for method_name in "${method_list[@]}"; do
    current=$((current + 1))
    echo ""
    echo "[$current/$TOTAL] $method_name"
    if ! run_method "$method_name"; then
        echo "ERROR: Training stopped after $method_name failed." >&2
        exit 1
    fi
done

echo ""
echo "========================================================"
echo "  All $TOTAL methods completed!"
echo "========================================================"
