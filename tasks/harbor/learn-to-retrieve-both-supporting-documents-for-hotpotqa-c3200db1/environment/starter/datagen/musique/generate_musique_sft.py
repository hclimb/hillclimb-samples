#!/usr/bin/env python3
"""
MuSiQue CoT SFT generation (stage 2 of 2). Runs on a TPU box.

Reads parquets from {HF_USERNAME}/musique-sft-base (stage 1), generates CoT
answers with Qwen3-4B over the FULL 20-paragraph context, filters by token
length and an LLM judge, uploads matching parquets to {HF_USERNAME}/musique-sft.

Modelled on datagen/generate_multihop_sft.py; four deliberate differences:

  0. Paragraphs are shown UNNUMBERED. See format_paragraphs — numbering them made
     the CoT cite "[Document 3]" in 99.6% of rows, references that point at
     nothing once the documents move into the memory bank.
  1. The prompt shows supporting AND distractor paragraphs, in MuSiQue's native
     (already-scrambled) order. The memory model has to discriminate gold from
     distractor at retrieval time, so the CoT it learns from should demonstrate
     that, not just read pre-filtered gold text.
  2. The judge sees answer_aliases. MuSiQue answers are entities with common
     surface variants; judging against the single canonical string rejects
     correct generations.
  3. Each row records hop_grounding — the fraction of MuSiQue's gold
     intermediate answers (from question_decomposition) that literally appear
     in the generated think block. The judge only checks the FINAL answer, so
     it happily passes a lucky shortcut that skipped a hop. This is stored,
     not filtered on by default: one generation pass is expensive, and the
     threshold can be applied to the parquets afterwards without regenerating.
     Check the logged histogram before picking one.

Cross-machine safe: lists completed parquets in the output repo at startup and
skips them. Accepts --parquets for datagen/persistent_tpu/orchestrator.py.

Usage:
    uv run python datagen/musique/generate_musique_sft.py
    uv run python datagen/musique/generate_musique_sft.py --dry-run
    uv run python datagen/musique/generate_musique_sft.py --parquets 0,1,2
"""

import asyncio
import argparse
import json
import logging
import os
import random
import re
import sys

import dotenv

dotenv.load_dotenv()

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from tqdm.asyncio import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))          # datagen/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))    # repo root, for evals/
from vllm_inference import VLLMInference

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ─── constants ────────────────────────────────────────────────────────────────
MODEL          = "Qwen/Qwen3-4B"
BASE_URL       = "http://localhost:8000/v1"
MAX_MODEL_LEN  = 8192   # 20 paragraphs ≈ 2.7k tokens of context; ample headroom
MAX_NUM_SEQS   = 256


def detect_tpu_chips(default: int = 8) -> int:
    """Count local TPU chips from /dev/vfio.

    generate_multihop_sft.py hardcodes 8, which silently mis-sizes tensor
    parallelism on anything that isn't a v6e-8 (tpu-v6e-vm is a 4-chip
    ct6e-standard-4t). Detect instead, --tp-size overrides.
    """
    try:
        return sum(1 for d in os.listdir("/dev/vfio") if d.isdigit()) or default
    except OSError:
        return default

_USER          = os.environ.get("HF_USERNAME", "")
INPUT_REPO     = f"{_USER}/musique-sft-base"
OUTPUT_REPO    = f"{_USER}/musique-sft"

# vLLM generation budget. Not 1024 (the value generate_multihop_sft.py uses): vLLM rejects a
# request when prompt + max_tokens > max_model_len, and MuSiQue's longest 20-paragraph contexts
# reach ~7.5k tokens, so 1024 pushes them past 8192 and the row is lost. Anything over
# MAX_THINK_ANS_TOKENS is filtered out regardless, so a 512 budget discards nothing we would
# have kept while leaving headroom for the long-context rows.
MAX_OUTPUT_TOKENS = 512

# Budget for think + answer. NOT 512, even though training seq_len is 512:
# _StreamingQAFilter (data/qa.py:105-122) drops any row whose FULL formatted
# sequence exceeds seq_len, and that sequence is
#   chat_template(question) + "<think>\n\n{think}</think>\n\n{answer}" + eos
# Template overhead ≈ 30 tok, question ≈ 25-40 tok, think wrapper ≈ 8 tok, so the
# real ceiling is ≈ 448. 400 leaves margin for long questions. Generating at 512
# would silently lose rows at training time, not here.
MAX_THINK_ANS_TOKENS = 400

MAX_CONCURRENCY = 256
LOCAL_DIR       = "./temp_musique_sft"

SCHEMA = pa.schema([
    ("id",               pa.string()),
    ("question",         pa.string()),
    ("answer",           pa.string()),   # MuSiQue ground truth — the training target
    ("generated_answer", pa.string()),   # model's paraphrase; kept for analysis only
    ("think",            pa.string()),
    ("pos_doc",          pa.list_(pa.string())),
    ("neg_doc",          pa.list_(pa.string())),
    ("hop_type",         pa.string()),
    ("hop_grounding",    pa.float32()),
])

# ─── prompts ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = "You are a helpful assistant."

ANSWER_PROMPT = """\
I am going to give you some additional information that might help you answer a question. Some of these documents are relevant and some are not.

Additional Information:
{formatted_paragraphs}

INSTRUCTION: Answering requires chaining facts together, where the answer to one step tells you what to look up next. Work through the steps, then give the answer.

HARD LIMIT: your reasoning must be UNDER 100 WORDS — about one short sentence per step. Name only the fact you took from each source; never quote or restate a source. Stop reasoning and answer the moment you have the answer.

Refer to a source by its bolded title, never by its number or position (there is no "first" or "third" document). Give the final answer as a short phrase — a name, date, or place — with no explanation.

QUESTION: {question}"""

JUDGE_PROMPT = """\
Question: {question}
Ground truth answer: {answer}{aliases_line}
Model answer: {generated_answer}

Does the model answer correctly answer the question, matching the ground truth (or any accepted alternative)? Answer with <verdict>YES</verdict> or <verdict>NO</verdict> only."""


# ─── helpers ──────────────────────────────────────────────────────────────────
def format_paragraphs(pos_doc: list[str], neg_doc: list[str], row_id: str) -> str:
    """Interleave gold and distractor paragraphs deterministically.

    Stage 1 splits MuSiQue's paragraph list into pos/neg, which destroys the
    original scrambled ordering and would leave all gold docs first — a
    positional shortcut the CoT could exploit. Re-scatter with a per-row seed
    derived from the id so the ordering is stable across reruns and resumes.

    Deliberately NOT numbered ("[Document 3]"). A first pass that numbered them
    produced CoT citing document indices in 99.6% of rows — but at training time
    the prompt holds only the question and the paragraphs live in the memory bank
    with no numbering, so those references point at nothing the model can ever
    see. Each paragraph already carries a `**Title**` header, which is real
    retrievable content and is the right thing for the reasoning to name.
    """
    docs = list(pos_doc) + list(neg_doc)
    random.Random(row_id).shuffle(docs)
    return "\n\n".join(docs)


def hop_grounding_score(think: str, decomposition_json: str) -> float:
    """Fraction of gold intermediate answers that appear verbatim in the think block.

    Substring matching, so it under-counts paraphrases — treat it as a floor on
    reasoning fidelity, not a precise measure.
    """
    try:
        steps = json.loads(decomposition_json)
    except Exception:
        return 0.0
    if not steps:
        return 0.0
    think_l = think.lower()
    # Skip blank intermediate answers: "" is a substring of everything, so counting them
    # would score an unrelated think block as fully grounded.
    golds = [a for a in (str(s.get("answer", "")).strip().lower() for s in steps) if a]
    if not golds:
        return 0.0
    return sum(1 for a in golds if a in think_l) / len(golds)


# ─── row processing ───────────────────────────────────────────────────────────
async def process_row(
    row: dict,
    vllm: VLLMInference,
    tokenizer,
    semaphore: asyncio.Semaphore,
    stats: dict,
    groundings: list,
) -> dict | None:
    """Generate CoT + answer for one row; apply token and judge filters."""
    async with semaphore:
        question = row["question"]
        ground_truth = row["answer"]
        aliases = list(row.get("answer_aliases") or [])
        formatted_paragraphs = format_paragraphs(row["pos_doc"], row["neg_doc"], row["id"])

        # Generation
        try:
            generated_answer, think = await vllm.async_chat(
                prompt=ANSWER_PROMPT.format(
                    formatted_paragraphs=formatted_paragraphs,
                    question=question,
                ),
                system=SYSTEM_PROMPT,
                max_completion_tokens=MAX_OUTPUT_TOKENS,
                temperature=0.6,
                thinking=True,
            )
        except Exception as e:
            err_str = str(e)
            if "max_tokens" in err_str and "too large" in err_str:
                stats["input_too_long"] += 1
            else:
                logging.warning(f"Generation error: {e}")
                stats["gen_error"] += 1
            return None

        think = (think or "").strip()
        generated_answer = generated_answer.strip()

        if not think:
            stats["no_think"] += 1
            return None

        # A response cut off right at </think> leaves think populated and content empty.
        # Only the judge would stand between that and an accepted row whose training target
        # is a think block with no answer after it.
        if not generated_answer:
            stats["empty_answer"] += 1
            return None

        # Token filter
        if len(tokenizer.encode(think + generated_answer)) > MAX_THINK_ANS_TOKENS:
            stats["token_filtered"] += 1
            return None

        # Judge filter
        aliases_line = f"\nAccepted alternatives: {'; '.join(aliases)}" if aliases else ""
        try:
            verdict_text, _ = await vllm.async_chat(
                prompt=JUDGE_PROMPT.format(
                    question=question,
                    answer=ground_truth,
                    aliases_line=aliases_line,
                    generated_answer=generated_answer,
                ),
                system=SYSTEM_PROMPT,
                max_completion_tokens=16,
                temperature=0.0,
                thinking=False,
            )
        except Exception as e:
            logging.warning(f"Judge error: {e}")
            stats["judge_error"] += 1
            return None

        verdict_match = re.search(r"<verdict>(YES|NO)</verdict>", verdict_text, re.IGNORECASE)
        if not verdict_match or verdict_match.group(1).upper() != "YES":
            stats["judge_rejected"] += 1
            return None

        grounding = hop_grounding_score(think, row["decomposition"])
        groundings.append(grounding)

        stats["accepted"] += 1
        return {
            "id":               row["id"],
            "question":         question,
            "answer":           ground_truth,
            "generated_answer": generated_answer,
            "think":            think,
            "pos_doc":          list(row["pos_doc"]),
            "neg_doc":          list(row["neg_doc"]),
            "hop_type":         row["hop_type"],
            "hop_grounding":    grounding,
        }


# ─── parquet processing ───────────────────────────────────────────────────────
async def process_parquet(
    parquet_filename: str,
    vllm: VLLMInference,
    tokenizer,
    api: HfApi,
    hf_token: str,
    dry_run: bool,
) -> None:
    logging.info(f"Downloading {parquet_filename}...")
    dl_path = hf_hub_download(
        repo_id=INPUT_REPO,
        filename=parquet_filename,
        repo_type="dataset",
        token=hf_token,
    )

    rows = pq.read_table(dl_path).to_pylist()
    logging.info(f"  {len(rows)} rows loaded.")

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    stats = {k: 0 for k in (
        "accepted", "token_filtered", "judge_rejected", "no_think", "empty_answer",
        "gen_error", "judge_error", "input_too_long",
    )}
    groundings: list[float] = []

    tasks = [process_row(row, vllm, tokenizer, semaphore, stats, groundings) for row in rows]
    results = await tqdm.gather(*tasks, desc=os.path.basename(parquet_filename))
    records = [r for r in results if r is not None]

    logging.info(
        f"  {parquet_filename}: {stats['accepted']} accepted | "
        f"{stats['token_filtered']} token-filtered | "
        f"{stats['judge_rejected']} judge-rejected | "
        f"{stats['no_think']} no-think | "
        f"{stats['empty_answer']} empty-answer | "
        f"{stats['input_too_long']} input-too-long | "
        f"{stats['gen_error']} gen-errors | "
        f"{stats['judge_error']} judge-errors"
    )

    # Histogram drives the hop_grounding threshold choice; see module docstring.
    if groundings:
        buckets = {"0.0": 0, "(0,0.5)": 0, "0.5-<1.0": 0, "1.0": 0}
        for g in groundings:
            if g == 0.0:      buckets["0.0"] += 1
            elif g < 0.5:     buckets["(0,0.5)"] += 1
            elif g < 1.0:     buckets["0.5-<1.0"] += 1
            else:             buckets["1.0"] += 1
        n = len(groundings)
        logging.info(
            "  hop_grounding: " + " | ".join(f"{k}={v} ({100*v/n:.0f}%)" for k, v in buckets.items())
        )

    if not records:
        logging.info("  No rows passed filters; skipping upload.")
        return

    output_basename = os.path.basename(parquet_filename)
    local_output = os.path.join(LOCAL_DIR, output_basename)
    os.makedirs(LOCAL_DIR, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records, schema=SCHEMA), local_output, compression="snappy")
    logging.info(f"  Wrote {len(records)} rows → {local_output}")

    if not dry_run:
        logging.info(f"  Uploading to {OUTPUT_REPO}/{parquet_filename}...")
        api.upload_file(
            path_or_fileobj=local_output,
            path_in_repo=parquet_filename,
            repo_id=OUTPUT_REPO,
            repo_type="dataset",
            commit_message=f"Add MuSiQue SFT data: {parquet_filename}",
            token=hf_token,
        )
        logging.info(f"  Uploaded {parquet_filename}.")

    os.remove(local_output)


# ─── main ─────────────────────────────────────────────────────────────────────
async def main():
    # Declared before the names are read as argparse defaults below — Python rejects a
    # `global` that follows any use of the name in the same function.
    global MAX_THINK_ANS_TOKENS, MAX_OUTPUT_TOKENS, MAX_MODEL_LEN

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Skip HF uploads")
    parser.add_argument(
        "--parquets",
        default=None,
        help="Comma-separated shard indices to process (orchestrator contract), e.g. '0,1,2'",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most N shards")
    parser.add_argument("--tp-size", type=int, default=None,
                        help="Tensor-parallel size (default: number of local TPU chips)")
    # Token budgets are flags so a seq_len change is an A/B, not a source edit. The think+answer
    # ceiling is set by the TRAINING seq_len: _StreamingQAFilter drops any row whose full
    # formatted sequence exceeds it, and the chat-template prefix costs ~33 tok + ~8 for the
    # think wrapper. So seq_len 512 -> ~400, seq_len 1024 -> ~950.
    parser.add_argument("--max-think-ans-tokens", type=int, default=MAX_THINK_ANS_TOKENS,
                        help="Reject think+answer longer than this (match the training seq_len)")
    parser.add_argument("--max-output-tokens", type=int, default=MAX_OUTPUT_TOKENS,
                        help="vLLM generation budget; must exceed --max-think-ans-tokens")
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN,
                        help="vLLM context. Must exceed longest prompt (~7.5k) + output budget")
    args = parser.parse_args()

    MAX_THINK_ANS_TOKENS = args.max_think_ans_tokens
    MAX_OUTPUT_TOKENS    = args.max_output_tokens
    MAX_MODEL_LEN        = args.max_model_len
    if MAX_OUTPUT_TOKENS <= MAX_THINK_ANS_TOKENS:
        raise SystemExit(
            f"--max-output-tokens ({MAX_OUTPUT_TOKENS}) must exceed --max-think-ans-tokens "
            f"({MAX_THINK_ANS_TOKENS}); otherwise every row truncates before it can be judged."
        )
    logging.info(
        f"Budgets: think+answer<={MAX_THINK_ANS_TOKENS}, output<={MAX_OUTPUT_TOKENS}, "
        f"model_len={MAX_MODEL_LEN}"
    )

    tp_size = args.tp_size or detect_tpu_chips()
    tpu_devices = ",".join(str(i) for i in range(tp_size))

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise ValueError("Set HF_TOKEN or HUGGINGFACE_TOKEN in your environment.")
    if not _USER:
        raise ValueError("Set HF_USERNAME in .env.")

    api = HfApi(token=hf_token)

    logging.info(f"Listing parquets in {INPUT_REPO}...")
    input_files = {
        f for f in api.list_repo_files(INPUT_REPO, repo_type="dataset")
        if f.endswith(".parquet")
    }
    logging.info(f"  {len(input_files)} parquets in input repo.")

    output_files: set[str] = set()
    if not args.dry_run:
        api.create_repo(OUTPUT_REPO, repo_type="dataset", exist_ok=True)
        try:
            output_files = {
                f for f in api.list_repo_files(OUTPUT_REPO, repo_type="dataset")
                if f.endswith(".parquet")
            }
            logging.info(f"  {len(output_files)} parquets already in output repo.")
        except Exception:
            logging.info("  Output repo empty or does not exist yet.")

    pending = sorted(input_files - output_files)

    if args.parquets:
        wanted = {int(x) for x in args.parquets.split(",") if x.strip()}
        pending = [f for f in pending if int(re.search(r"(\d+)\.parquet$", f).group(1)) in wanted]
        logging.info(f"  --parquets {sorted(wanted)} → {len(pending)} shards selected.")
    if args.limit is not None:
        pending = pending[: args.limit]

    logging.info(f"  {len(pending)} parquets to process.")
    if not pending:
        logging.info("Nothing to do.")
        return

    # A killed run leaves EngineCore workers holding /dev/vfio/*, and the next launch dies on
    # "Device or resource busy" only after downloading the model. pkill on "vllm serve" misses
    # them — it matches the parent, not the workers. evals/vllm.py already solves this properly
    # (fuser over /dev/accel* + /dev/vfio/*, then SIGKILL), so reuse it.
    from evals.vllm import VLLMInference as _EvalVLLM
    freed = _EvalVLLM.free_tpu_devices()
    if freed:
        logging.info(f"Freed {freed} process(es) still holding TPU devices.")

    logging.info(f"Starting {MODEL} on {tp_size} TPU chips (devices {tpu_devices})...")
    server_proc = VLLMInference.start_server(
        model=MODEL,
        base_url=BASE_URL,
        tensor_parallel_size=tp_size,
        tpu_visible_devices=tpu_devices,
        download_dir="/dev/shm",
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
    )
    try:
        VLLMInference.wait_for_server(BASE_URL, timeout=600)
    except TimeoutError:
        server_proc.terminate()
        raise

    vllm = VLLMInference(model=MODEL, base_url=BASE_URL)

    logging.info("Loading tokenizer for token counting...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    try:
        for i, parquet_filename in enumerate(pending):
            logging.info(f"[{i+1}/{len(pending)}] {parquet_filename}")
            await process_parquet(parquet_filename, vllm, tokenizer, api, hf_token, args.dry_run)
    finally:
        logging.info("Terminating vLLM server...")
        server_proc.terminate()
        server_proc.wait()
        logging.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
