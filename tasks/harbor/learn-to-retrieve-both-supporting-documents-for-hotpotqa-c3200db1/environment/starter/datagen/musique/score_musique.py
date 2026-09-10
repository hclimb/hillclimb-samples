"""
Score MuSiQue predictions using the official evaluate_v1.0.py evaluator.

Takes the JSON output from the gen_embed evaluator and converts it to the
format expected by evaluate_v1.0.py, then runs scoring.

Matches predictions to gold by position (both ordered the same way since
gen_embed streams validation split without filtering).

Usage:
    uv run python datagen/musique/score_musique.py \
        --results path/to/gen_embed_musique_results.json \
        --gold datagen/musique/musique_data/extracted/musique_ans_v1.0_dev.jsonl
"""

import argparse
import json
import os
import subprocess
import sys


def load_gen_embed_results(results_path: str) -> list:
    with open(results_path) as f:
        data = json.load(f)
    return data["samples"]


def build_predictions(samples: list, gold_rows: list) -> list:
    if len(samples) != len(gold_rows):
        print(
            f"[WARNING] predictions ({len(samples)}) and gold ({len(gold_rows)}) "
            f"counts differ. Matching up to min({len(samples)}, {len(gold_rows)})."
        )

    predictions = []
    for sample, gold in zip(samples, gold_rows):
        predictions.append({
            "id": gold["id"],
            "predicted_answer": sample.get("generated_answer", sample["generated"]),
            "predicted_support_idxs": [],
            "predicted_answerable": True,
        })
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to gen_embed output JSON")
    parser.add_argument("--gold", required=True, help="Path to musique_ans_v1.0_dev.jsonl")
    parser.add_argument("--output-dir", default=None, help="Where to save predictions.jsonl")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    evaluator_repo = os.path.join(script_dir, "musique_repo", "evaluate_v1.0.py")
    if not os.path.exists(evaluator_repo):
        raise FileNotFoundError(
            f"Official evaluator not found at {evaluator_repo}. "
            "Run: git clone --depth=1 https://github.com/StonyBrookNLP/musique.git datagen/musique/musique_repo"
        )

    print("Loading gen_embed results...")
    samples = load_gen_embed_results(args.results)
    print(f"  {len(samples)} predictions")

    print("Loading gold answers...")
    gold_by_id = {}
    with open(args.gold) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                gold_by_id[row["id"]] = row
    print(f"  {len(gold_by_id)} gold rows loaded")

    answerable_gold = [r for r in gold_by_id.values() if r.get("answerable", False)]
    predictions = build_predictions(samples, answerable_gold)

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.results))
    pred_path = os.path.join(output_dir, "musique_predictions.jsonl")
    gold_subset_path = os.path.join(output_dir, "musique_gold_subset.jsonl")

    with open(pred_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")
    print(f"Saved {len(predictions)} predictions to {pred_path}")

    pred_ids = [p["id"] for p in predictions]
    with open(gold_subset_path, "w") as f:
        for pid in pred_ids:
            f.write(json.dumps(gold_by_id[pid]) + "\n")
    print(f"Saved gold subset to {gold_subset_path}")

    print("\nRunning official MuSiQue evaluator...")
    result = subprocess.run(
        [sys.executable, evaluator_repo, pred_path, gold_subset_path],
        capture_output=False,
        cwd=os.path.dirname(evaluator_repo),
    )
    if result.returncode != 0:
        print(f"Evaluator exited with code {result.returncode}")


if __name__ == "__main__":
    main()
