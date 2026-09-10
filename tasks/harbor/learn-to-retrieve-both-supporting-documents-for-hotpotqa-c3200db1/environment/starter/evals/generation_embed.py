import jax
import jax.numpy as jnp
import numpy as np
import json
import wandb
from functools import partial
from tqdm import tqdm

from jax.sharding import PartitionSpec as P

from .base import Evaluator
from .utils import split_thinking
from inference import _generate_tokens
from losses.doc_access_acc import compute_doc_access_acc
from utils import unfreeze_dict


def _per_example_pos_slot_mass(aux_data, loss_mask, input_mask):
    """Per-example softmax weight mass on positive-doc slots (answer-region proxy), averaged
    over memory layers. Returns np array [B] or None. Same positive-doc join as doc_access_acc /
    mem_pos_weight_mass, kept per-example so it can be split by the LLM-judge verdict downstream
    (grounding_experiments_plan.md eval diagnostic #2)."""
    idx_list = aux_data.get("mem_top_k_indices")
    prob_list = aux_data.get("mem_top_k_probs")
    if not idx_list or not prob_list:
        return None
    docs_mask = input_mask["docs_mask"]
    num_docs_total, raw_doc_len = docs_mask.shape
    doc_len = aux_data.get("effective_doc_len", raw_doc_len)
    eff_mask = aux_data.get("effective_mem_mask", None)
    validity = eff_mask if eff_mask is not None else docs_mask.flatten()
    acc, nL = None, 0
    for li, indices in enumerate(idx_list):
        if li >= len(prob_list):
            break
        B, H, S, K = indices.shape
        rdi = indices // doc_len
        pm = input_mask.get("pos_doc_mask")
        if pm is None:
            pm = jnp.ones((B, 1), dtype=jnp.float32)
        Bl, _ = pm.shape
        be = jnp.eye(Bl, dtype=pm.dtype)
        pdg = (be[:, :, None] * pm[None, :, :]).reshape(Bl, num_docs_total)
        bi = jnp.arange(B)[:, None, None, None]
        is_pos = pdg.at[bi, rdi].get(out_sharding=P('data', 'model', None, None)) > 0
        is_valid_tok = validity.at[indices].get(out_sharding=P('data', 'model', None, None)) == 1
        is_active = loss_mask[:, None, :, None].astype(jnp.bool_)
        is_valid = is_valid_tok & is_active
        probs = prob_list[li].astype(jnp.float32)
        num = (probs * (is_pos & is_valid).astype(jnp.float32)).sum(axis=(1, 2, 3))
        den = (probs * is_valid.astype(jnp.float32)).sum(axis=(1, 2, 3)) + 1e-6
        m = num / den
        acc = m if acc is None else acc + m
        nL += 1
    return None if acc is None else np.array(acc / nL)


class GenerationEmbedEvaluator(Evaluator):
    """
    Generation evaluator for qwen3_mem_embed-style models.

    Unlike GenerationEvaluator (which feeds raw text through generate()), this
    evaluator:
      1. Pulls tokenised batches — with docs — directly from dataset.generator().
      2. Pre-embeds the doc batch once per forward pass (outside JIT).
      3. Injects the resulting memory bank into model.weights under the
         "main_model.mem_k / mem_v / mem_mask" keys so that model.forward
         skips re-embedding inside the JIT-compiled decode loop.
      4. Calls the existing _generate_tokens (JIT-compiled) for fast decoding.

    The dataset must be configured with provide_docs=True (e.g. BioR with
    doc_type="bio" or doc_type="qa") so that generator() yields the
    {"batch": ..., "docs": ...} dict structure.
    """

    def evaluate(self, model, dataset, step=None, aux_loss_config=None, **kwargs):
        if jax.process_index() == 0:
            print("Starting Generation Embed Evaluation...")

        aux_loss_cfg_dict = unfreeze_dict(aux_loss_config) if aux_loss_config is not None else {}
        compute_acc = bool(aux_loss_cfg_dict.get('doc_access_acc', {}).get('enabled', False))
        # Answer-slot diagnostic: attach per-example positive-slot weight mass to each result so
        # it can be split by the LLM-judge verdict downstream. Off by default (eval.answer_slot_diag).
        answer_slot_diag = bool(self.cfg.get("answer_slot_diag", False))
        
        # ----------------------------------------------------------------
        # 1. Import model-specific helpers
        # ----------------------------------------------------------------
        from models.qwen3_mem_embed import embed_forward
        from models.utils import split_weights

        # ----------------------------------------------------------------
        # 2. Validate that the dataset will provide docs
        # ----------------------------------------------------------------
        if not getattr(dataset, "provide_docs", False):
            raise ValueError(
                "GenerationEmbedEvaluator requires a dataset configured with "
                "provide_docs=True so that generator() yields document tokens."
            )

        # ----------------------------------------------------------------
        # 4. Split model weights once; build a stateless embed callable
        #    that will be called (eagerly) once per batch.
        # ----------------------------------------------------------------
        # For distill models, unwrap student weights (student.main_model.* / student.embed_model.*)
        weights = model.weights
        cfg = model.cfg
        if "student" in cfg:
            student_w, = split_weights(weights, ["student"])
            weights = student_w
            cfg = cfg["student"]

        # Stage 2: if a separate value_model is present, memory VALUES must come from it
        # (keys from the embed model) — otherwise oracle eval of Stage 2/3 checkpoints would
        # silently read values off the key model. Mirrors qwen3_mem_embed.forward.
        value_present = "value_model" in cfg
        if value_present:
            from models.qwen3_mem_embed import value_forward
            _main_w, embed_w, value_w = split_weights(weights, ["main_model", "embed_model", "value_model"])
            value_cfg = cfg["value_model"]
        else:
            _main_w, embed_w = split_weights(weights, ["main_model", "embed_model"])
        embed_cfg = cfg["embed_model"]

        # embed_fn(docs_jax, dmask_jax) -> (mem_k, mem_v, mem_mask, eff_doc_len)
        def embed_fn(docs, dmask):
            mem_k, mem_v_e, mem_mask, eff = embed_forward(embed_cfg, docs, embed_w, dmask)
            mem_v = value_forward(value_cfg, docs, value_w, dmask) if value_present else mem_v_e
            return mem_k, mem_v, mem_mask, eff

        # For distill models, build a student-only forward for generation
        # (the distill forward expects teacher_batch which isn't available during generation)
        is_distill = "student" in model.cfg
        if is_distill:
            from models.qwen3_mem_embed import forward as mem_embed_fwd
            gen_forward = partial(mem_embed_fwd, cfg)
            gen_weights = weights
        else:
            gen_forward = model.forward
            gen_weights = None  # will use model.weights below

        tokenizer = model.tokenizer

        # ----------------------------------------------------------------
        # 5. Optionally enable chunked retrieval via model cfg
        # ----------------------------------------------------------------
        lookup_chunk_size = self.cfg.get("lookup_chunk_size", None)
        reset_chunk_size = False
        if lookup_chunk_size is not None:
            if jax.process_index() == 0:
                print(f"  Enabling chunked retrieval (lookup_chunk_size={lookup_chunk_size})")
            if model.cfg.get("mem_lookup_chunk_size", None) is None:
                reset_chunk_size = True
            model.cfg['main_model']['mem_lookup_chunk_size'] = lookup_chunk_size

        # ----------------------------------------------------------------
        # 6. Iterate over the dataset in pre-batched form
        # ----------------------------------------------------------------
        iterator = dataset.generator(num_epochs=1)

        num_samples = self.cfg.get("num_samples", None)

        if compute_acc or answer_slot_diag:
            @partial(jax.jit, static_argnames=("forward",))
            def _prefill_aux_fn(forward, params, prompt_tokens, pad_mask):
                return forward(prompt_tokens, params, pad_mask=pad_mask, collect_aux=True).aux

        results = []
        acc_scores = []
        telemetry_batches = []
        count   = 0
        pbar    = tqdm(total=num_samples, desc="Generating (embed)")

        for batch_tokens, batch_masks in iterator:
            if num_samples is not None and count >= num_samples:
                break

            # -- Unpack batch -------------------------------------------
            if not isinstance(batch_tokens, dict) or "docs" not in batch_tokens:
                raise ValueError(
                    "Expected batch_tokens to be a dict with 'batch' and 'docs' keys. "
                    "Make sure the dataset has provide_docs=True."
                )

            raw_batch  = np.array(batch_tokens["batch"])         # (B, T)
            raw_docs   = np.array(batch_tokens["docs"])          # (B*M, doc_seq_len) or (B, doc_len)
            batch_mask = np.array(batch_masks["batch_mask"])     # (B, T)
            loss_mask  = np.array(batch_masks["loss_mask"])      # (B, T)
            docs_mask  = np.array(batch_masks["docs_mask"])      # (B*M, doc_seq_len) or (B, doc_len)
            pos_doc_mask_np = np.array(batch_masks["pos_doc_mask"]) if "pos_doc_mask" in batch_masks else None

            B, T = raw_batch.shape

            # -- Determine prompt / answer split per example ------------
            # Prompt  = positions where batch_mask==1 AND loss_mask==0
            # Answer  = positions where loss_mask==1
            prompt_ends    = []
            gt_answer_ids  = []

            for i in range(B):
                ans_pos = np.where(loss_mask[i] > 0)[0]
                pe = int(ans_pos[0]) if len(ans_pos) > 0 else T
                prompt_ends.append(pe)
                gt_answer_ids.append(
                    raw_batch[i, ans_pos].tolist() if len(ans_pos) > 0 else []
                )

            max_prompt_len = max(prompt_ends)

            # Left-pad (right-align) prompts so that position -1 is always the
            # last real token. _generate_tokens takes logits[:, -1, :] after
            # prefill to seed the first generated token, so shorter prompts
            # must have their padding at the front, not the back.
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            prompt_tokens   = np.full((B, max_prompt_len), pad_id, dtype=raw_batch.dtype)
            prompt_pad_mask = np.zeros((B, max_prompt_len), dtype=np.bool_)

            for i in range(B):
                pe = prompt_ends[i]
                offset = max_prompt_len - pe
                prompt_tokens[i,   offset:] = raw_batch[i, :pe]
                prompt_pad_mask[i, offset:] = batch_mask[i, :pe].astype(np.bool_)

            # -- Pre-embed docs (eager, not JIT) ------------------------
            docs_jax  = jax.device_put(
                jnp.array(raw_docs),
                jax.sharding.PartitionSpec("data", None)
            )
            dmask_jax = jax.device_put(
                jnp.array(docs_mask.astype(np.bool_)),
                jax.sharding.PartitionSpec("data", None)
            )
            mem_k, mem_v, mem_mask_flat, eff_doc_len = embed_fn(docs_jax, dmask_jax)
            
            # Replicate memory fully across devices so it does not conflict
            # with the data-sharded batch inside main_forward.
            #every TPU has all the memk,memv now
            mem_k_rep    = jax.device_put(np.array(jax.experimental.multihost_utils.process_allgather(mem_k, tiled=True)),         jax.sharding.PartitionSpec())
            mem_v_rep    = jax.device_put(np.array(jax.experimental.multihost_utils.process_allgather(mem_v, tiled=True)),         jax.sharding.PartitionSpec())
            mem_mask_rep = jax.device_put(np.array(jax.experimental.multihost_utils.process_allgather(mem_mask_flat, tiled=True)), jax.sharding.PartitionSpec())

            # -- Inject pre-embedded memory into a shallow copy of params -
            # qwen3_mem_embed.forward checks main_weights["mem_k"] and
            # skips doc embedding when it is non-empty.
            # For distill models, use student weights directly (gen_weights);
            # for standard models, use model.weights.
            base_weights = gen_weights if is_distill else model.weights
            params_with_mem = dict(base_weights)
            params_with_mem["main_model.mem_k"]    = mem_k_rep
            params_with_mem["main_model.mem_v"]    = mem_v_rep
            params_with_mem["main_model.mem_mask"] = mem_mask_rep

            # -- Shard prompt input -------------------------------------
            # Pad batch to be divisible by the data-axis size so that
            # device_put with PartitionSpec("data", None) doesn't fail
            # when B < n_data (e.g. batch_size=1 with 8 data devices).
            base_weights_for_mesh = gen_weights if is_distill else model.weights
            some_w = next((v for v in base_weights_for_mesh.values() if hasattr(getattr(v, 'sharding', None), 'mesh')), None)
            n_data = some_w.sharding.mesh.shape.get("data", 1) if some_w is not None else 1
            B_pad = int(np.ceil(B / n_data)) * n_data if n_data > 1 else B
            if B_pad > B:
                prompt_tokens   = np.concatenate([prompt_tokens,   np.full((B_pad - B, prompt_tokens.shape[1]),   pad_id,    dtype=prompt_tokens.dtype)],   axis=0)
                prompt_pad_mask = np.concatenate([prompt_pad_mask, np.zeros((B_pad - B, prompt_pad_mask.shape[1]), dtype=prompt_pad_mask.dtype)], axis=0)

            prompt_jax = jax.device_put(
                jnp.array(prompt_tokens),
                jax.sharding.PartitionSpec("data", None)
            )
            pmask_jax = jax.device_put(
                jnp.array(prompt_pad_mask),
                jax.sharding.PartitionSpec("data", None)
            )

            # NOTE: aux metrics (doc_access_acc, mem telemetry, answer-slot mass) are
            # computed AFTER generation over a [prompt | generated-answer] forward masked
            # to the answer span — see the post-generation block below. This measures
            # retrieval at the SAME positions training does (answer tokens with the
            # model's own generated context) instead of at the prompt prefill, so the
            # eval aux numbers are comparable to the training-time aux numbers.
            batch_pos_mass = None

            # -- JIT-compiled generation --------------------------------
            # For distill models, gen_forward is the student's mem_embed forward;
            # for standard models, it's model.forward.
            gen_tokens = _generate_tokens(
                gen_forward,
                model.init_kv,
                params_with_mem,
                prompt_jax,
                self.cfg.max_new_tokens,
                pad_mask=pmask_jax,
                temperature=self.cfg.temperature,
                top_k=self.cfg.get("top_k", 20),
                top_p=self.cfg.get("top_p", 0.8),
            )

            gen_tokens_np = np.array(jax.experimental.multihost_utils.process_allgather(gen_tokens, tiled=True))[:B]  # (B, max_new_tokens) — drop padding rows

            # -- aux metrics on GENERATED-ANSWER positions (mode B) -----------------------
            # A single causal forward over [prompt | generated] reproduces each decode
            # step's query exactly (same own-generated input tokens + causal mask), so aux
            # computed here == accumulating aux over decode steps and averaging. We mask to
            # the generated answer span (before first EOS) → generic over ANY aux metric
            # derived from aux_data (doc_access_acc, all mem telemetry, answer-slot mass).
            if compute_acc or answer_slot_diag:
                try:
                    G = gen_tokens_np.shape[1]
                    ans_content = np.zeros((B, G), dtype=np.bool_)   # generated tokens before first EOS
                    for i in range(B):
                        eosp = np.where(gen_tokens_np[i] == tokenizer.eos_token_id)[0]
                        end = int(eosp[0]) if len(eosp) > 0 else G
                        ans_content[i, :end] = True
                    p_tok = prompt_tokens[:B]; p_pad = prompt_pad_mask[:B]
                    full_tokens = np.concatenate([p_tok, gen_tokens_np], axis=1)                       # (B, P+G)
                    full_pad    = np.concatenate([p_pad, ans_content], axis=1)                         # attend prompt + answer content
                    answer_mask = np.concatenate([np.zeros_like(p_pad), ans_content], axis=1).astype(np.float32)  # aux averaged over answer span only
                    if B_pad > B:   # pad rows for data-axis sharding (mirrors the prompt handling)
                        pr = B_pad - B
                        full_tokens = np.concatenate([full_tokens, np.full((pr, full_tokens.shape[1]), pad_id, dtype=full_tokens.dtype)], axis=0)
                        full_pad    = np.concatenate([full_pad,    np.zeros((pr, full_pad.shape[1]),   dtype=full_pad.dtype)], axis=0)
                    full_jax = jax.device_put(jnp.array(full_tokens), jax.sharding.PartitionSpec("data", None))
                    fpad_jax = jax.device_put(jnp.array(full_pad),    jax.sharding.PartitionSpec("data", None))
                    aux_data = _prefill_aux_fn(gen_forward, params_with_mem, full_jax, fpad_jax)
                    if aux_data is not None:
                        aux_data['effective_doc_len'] = int(eff_doc_len)
                        aux_data['effective_mem_mask'] = mem_mask_rep
                        input_mask_acc = {"docs_mask": jnp.array(docs_mask)}
                        if pos_doc_mask_np is not None:
                            input_mask_acc["pos_doc_mask"] = jnp.array(pos_doc_mask_np)
                        amask = jnp.array(answer_mask)
                        if compute_acc:
                            acc = float(compute_doc_access_acc(aux_data, amask, input_mask_acc, None))
                            acc_scores.append(acc)
                            try:
                                from losses.mem_telemetry import collect_eval_telemetry
                                tel = collect_eval_telemetry(aux_data, amask, input_mask_acc)
                                if tel:
                                    telemetry_batches.append(tel)
                            except Exception as _te:
                                if jax.process_index() == 0:
                                    print(f"  [telemetry] skipping batch: {_te}")
                        if answer_slot_diag:
                            batch_pos_mass = _per_example_pos_slot_mass(aux_data, amask, input_mask_acc)
                    del full_jax, fpad_jax
                except Exception as e:
                    if jax.process_index() == 0:
                        print(f"  [aux/doc_access_acc] skipping batch (OOM or error): {e}")

            # Free all device buffers from this batch before next iteration.
            # Deleting dict keys alone is not enough — the named variables below
            # keep the JAX buffers alive on HBM via Python reference counting.
            del params_with_mem["main_model.mem_k"]
            del params_with_mem["main_model.mem_v"]
            del params_with_mem["main_model.mem_mask"]
            del mem_k_rep, mem_v_rep, mem_mask_rep   # replicated memory bank
            del mem_k, mem_v, mem_mask_flat           # sharded outputs of embed_fn
            del docs_jax, dmask_jax                   # doc input tensors
            del prompt_jax, pmask_jax                 # prompt input tensors
            del gen_tokens                            # generated token buffer
            jax.effects_barrier()

            actual = min(B, num_samples - count) if num_samples is not None else B
            for i in range(actual):
                # Decode doc(s) — handle multi-doc (streaming_qa) vs single-doc (BioR)
                if pos_doc_mask_np is not None:
                    M = pos_doc_mask_np.shape[1]
                    raw_docs_3d = raw_docs.reshape(B, M, -1)
                    dm_3d = docs_mask.reshape(B, M, -1)
                    doc_texts = []
                    for d in range(M):
                        if pos_doc_mask_np[i, d]:
                            idx = np.where(dm_3d[i, d] > 0)[0]
                            doc_texts.append(tokenizer.decode(raw_docs_3d[i, d, idx], skip_special_tokens=True))
                    doc_str = " || ".join(doc_texts) if doc_texts else ""
                else:
                    doc_indices = np.where(docs_mask[i] > 0)[0]
                    doc_str = tokenizer.decode(raw_docs[i, doc_indices], skip_special_tokens=True)

                # Truncate at first EOS so post-EOS garbage from the fixed-length
                # fori_loop isn't included in the decoded output.
                eos_positions = np.where(gen_tokens_np[i] == tokenizer.eos_token_id)[0]
                gen_i = gen_tokens_np[i, :eos_positions[0]] if len(eos_positions) > 0 else gen_tokens_np[i]

                generated = tokenizer.decode(gen_i, skip_special_tokens=True)
                thinking, generated_answer = split_thinking(generated)
                res = {
                    "prompt":           tokenizer.decode(raw_batch[i, :prompt_ends[i]], skip_special_tokens=False),
                    "generated":        generated,
                    "thinking":         thinking,
                    "generated_answer": generated_answer,
                    "ground_truth":     tokenizer.decode(gt_answer_ids[i], skip_special_tokens=True),
                    "doc":              doc_str,
                }
                if batch_pos_mass is not None and i < len(batch_pos_mass):
                    res["pos_slot_weight_mass"] = float(batch_pos_mass[i])
                results.append(res)

            count += actual
            pbar.update(actual)

        pbar.close()

        if reset_chunk_size:
            model.cfg['main_model']['mem_lookup_chunk_size'] = None

        if self.cfg.output_file:
            if jax.process_index() == 0:
                output_path = self._get_output_path(step, self.cfg.output_file)
                file_metrics = {}
                if acc_scores:
                    file_metrics["doc_access_acc"] = float(np.mean(acc_scores))
                if telemetry_batches:
                    for tk in set().union(*telemetry_batches):
                        vals = [b[tk] for b in telemetry_batches if tk in b]
                        if vals:
                            file_metrics[tk] = float(np.mean(vals))
                with open(output_path, "w") as f:
                    json.dump({"metrics": file_metrics, "samples": results}, f, indent=2)
                print(f"Saved generations to {output_path}")
                if wandb.run is not None:
                    artifact_name = f"{wandb.run.id}-eval-{self.key}-step-{step}-results" if step is not None else f"{wandb.run.id}-eval-{self.key}-results"
                    artifact = wandb.Artifact(name=artifact_name, type="evaluation_results")
                    artifact.add_file(output_path)
                    wandb.log_artifact(artifact)

        inference_metrics = {"generated_count": count}
        if acc_scores:
            inference_metrics["doc_access_acc"] = float(np.mean(acc_scores))
        return inference_metrics
