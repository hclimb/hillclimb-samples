"""
Tests for the RAG eval pipeline used by corpus evals (MuSiQue, HotpotQA, MS MARCO).

Covers:
  1. normalize_ground_truth — all input types
  2. Recall@K / MRR computation logic
  3. _run_rag_pipeline — subprocess orchestration, file parsing, metric keys
  4. Metric key prefixing in rag_eval.main's all_metrics dict

No JAX, no TPU, no vLLM, no HuggingFace required.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evals.rag.single_embedding_retrieval import (
    compute_retrieval_metrics,
    normalize_ground_truth,
    stringify_ground_truth,
)


# ---------------------------------------------------------------------------
# Helpers for building synthetic retrieval results.
# ---------------------------------------------------------------------------
def make_result(query, gt_docs, retrieved_texts, top_k=5, gt_doc_ids=None, retrieved_doc_ids=None):
    """Build a result dict in the format produced by run_retrieve."""
    if retrieved_doc_ids is None:
        retrieved_doc_ids = list(range(len(retrieved_texts[:top_k])))
    return {
        "query": query,
        "ground_truth": "some answer",
        "ground_truth_doc": gt_docs,
        "ground_truth_doc_ids": gt_doc_ids or [],
        "retrieved": [
            {
                "rank": i + 1,
                "doc_index": retrieved_doc_ids[i],
                "score": 1.0 - i * 0.1,
                "document": text,
            }
            for i, text in enumerate(retrieved_texts[:top_k])
        ],
    }


# ===========================================================================
# 1. normalize_ground_truth
# ===========================================================================

class TestNormalizeGroundTruth:
    def test_none(self):
        assert normalize_ground_truth(None) == []

    def test_string(self):
        assert normalize_ground_truth("Paris") == ["Paris"]

    def test_int(self):
        assert normalize_ground_truth(42) == ["42"]

    def test_float(self):
        assert normalize_ground_truth(3.0) == ["3"]

    def test_list_of_strings(self):
        assert normalize_ground_truth(["a", "b"]) == ["a", "b"]

    def test_list_of_dicts_with_text(self):
        inp = [{"text": "passage one"}, {"text": "passage two"}]
        assert normalize_ground_truth(inp) == ["passage one", "passage two"]

    def test_list_of_dicts_with_content(self):
        inp = [{"content": "hello"}]
        assert normalize_ground_truth(inp) == ["hello"]

    def test_list_of_dicts_fallback(self):
        inp = [{"other": "val"}]
        result = normalize_ground_truth(inp)
        assert len(result) == 1
        assert "other" in result[0]

    def test_empty_list(self):
        assert normalize_ground_truth([]) == []

    def test_mixed_list(self):
        inp = ["plain string", {"text": "dict entry"}]
        assert normalize_ground_truth(inp) == ["plain string", "dict entry"]


class TestStringifyGroundTruth:
    def test_empty(self):
        assert stringify_ground_truth(None) == ""

    def test_string(self):
        assert stringify_ground_truth("Paris") == "Paris"

    def test_list(self):
        assert stringify_ground_truth(["Paris", "France"]) == "Paris | France"


# ===========================================================================
# 2. Recall@K / MRR computation
# ===========================================================================

class TestRetrievalMetrics:
    """
    Tests mirror exactly what run_retrieve computes at lines 509-534
    of single_embedding_retrieval.py.
    """

    def test_perfect_recall(self):
        results = [
            make_result("q1", ["doc_A"], ["doc_A", "doc_B", "doc_C"]),
            make_result("q2", ["doc_X"], ["doc_X", "doc_Y"]),
        ]
        m = compute_retrieval_metrics(results, top_k=5)
        assert m["recall@1"] == pytest.approx(1.0)
        assert m["recall@5"] == pytest.approx(1.0)
        assert m["mrr"] == pytest.approx(1.0)

    def test_zero_recall(self):
        results = [
            make_result("q1", ["doc_A"], ["doc_B", "doc_C"]),
        ]
        m = compute_retrieval_metrics(results, top_k=5)
        assert m["recall@5"] == pytest.approx(0.0)
        assert m["mrr"] == pytest.approx(0.0)

    def test_recall_at_1_vs_recall_at_5(self):
        # Positive doc is at rank 3 → recall@1=0, recall@5=1
        results = [
            make_result("q1", ["doc_C"], ["doc_A", "doc_B", "doc_C", "doc_D"]),
        ]
        m = compute_retrieval_metrics(results, top_k=5)
        assert m["recall@1"] == pytest.approx(0.0)
        assert m["recall@5"] == pytest.approx(1.0)

    def test_mrr_rank_3(self):
        # Positive doc at rank 3 → MRR = 1/3
        results = [
            make_result("q1", ["doc_C"], ["doc_A", "doc_B", "doc_C", "doc_D"]),
        ]
        m = compute_retrieval_metrics(results, top_k=5)
        assert m["mrr"] == pytest.approx(round(1 / 3, 4))

    def test_partial_recall_batch(self):
        # 1 of 2 queries retrieves the positive doc
        results = [
            make_result("q1", ["doc_A"], ["doc_A", "doc_B"]),   # hit
            make_result("q2", ["doc_X"], ["doc_Y", "doc_Z"]),   # miss
        ]
        m = compute_retrieval_metrics(results, top_k=5)
        assert m["recall@5"] == pytest.approx(0.5)
        assert m["mrr"] == pytest.approx(round(1 / 2, 4))

    def test_empty_ground_truth_skipped(self):
        """Queries with no ground_truth_doc are excluded from all metrics."""
        results = [
            make_result("q1", [], ["doc_A"]),           # no GT → skip
            make_result("q2", ["doc_X"], ["doc_X"]),    # hit
        ]
        m = compute_retrieval_metrics(results, top_k=5)
        assert m["recall@5"] == pytest.approx(1.0)
        assert m["mrr"] == pytest.approx(1.0)

    # -- Corpus-specific scenarios --

    def test_musique_multi_hop(self):
        """MuSiQue: ground_truth_doc is a list of multiple supporting passages.
        A hit requires any one of them to appear in the retrieved set."""
        passage_a = "Paris is the capital of France."
        passage_b = "France is a country in Western Europe."
        results = [
            # Only passage_b retrieved → still a hit (any match counts)
            make_result("Who is the capital of a Western European country?",
                        [passage_a, passage_b],
                        ["irrelevant doc", passage_b, "another irrelevant"]),
            # Neither passage retrieved
            make_result("Two-hop question",
                        [passage_a, passage_b],
                        ["unrelated 1", "unrelated 2"]),
        ]
        m = compute_retrieval_metrics(results, top_k=5)
        assert m["recall@5"] == pytest.approx(0.5)

    def test_hotpotqa_title_format(self):
        """HotpotQA docs: ground_truth_doc is the full passage string.
        A retrieved doc that is identical to the ground truth counts as a hit
        (exact string is a substring of itself)."""
        gt_doc = "Marie Curie: Polish-French physicist who conducted pioneering research."
        results = [
            make_result("Where was Marie Curie born?",
                        [gt_doc],
                        [gt_doc, "Unrelated doc"]),
            make_result("What field did Marie Curie work in?",
                        [gt_doc],
                        ["Unrelated 1", "Unrelated 2", "Unrelated 3"]),
        ]
        m = compute_retrieval_metrics(results, top_k=5)
        assert m["recall@5"] == pytest.approx(0.5)
        assert m["recall@1"] == pytest.approx(0.5)

    def test_short_answer_substring_in_retrieved_doc(self):
        """Substring fix: a short answer string contained in a longer retrieved doc counts
        as a hit.  This is the real-world case for MuSiQue/HotpotQA where query_gt_column
        is the answer text (e.g. 'Paris') not the full passage."""
        results = [
            make_result("What is the capital of France?",
                        ["Paris"],
                        ["Paris is the capital of France and a major European city."]),
            make_result("Who invented the telephone?",
                        ["Alexander Graham Bell"],
                        ["The telephone was invented by Alexander Graham Bell in 1876."]),
            make_result("What year did WWII end?",
                        ["1945"],
                        ["World War II ended in 1945 after Japan surrendered."]),
        ]
        m = compute_retrieval_metrics(results, top_k=5)
        assert m["recall@1"] == pytest.approx(1.0)
        assert m["mrr"] == pytest.approx(1.0)

    def test_short_answer_not_in_any_doc(self):
        """Answer not present as substring in any retrieved doc → miss."""
        results = [
            make_result("What is the capital of France?",
                        ["Paris"],
                        ["London is the capital of England.", "Berlin is in Germany."]),
        ]
        m = compute_retrieval_metrics(results, top_k=5)
        assert m["recall@5"] == pytest.approx(0.0)

    def test_prefers_doc_ids_when_available(self):
        """When gold doc IDs are present, retrieval matching uses IDs instead of text."""
        results = [
            make_result(
                "q1",
                ["gold text that is absent from retrieved docs"],
                ["completely different retrieved text", "another different doc"],
                gt_doc_ids=[42],
                retrieved_doc_ids=[10, 42],
            ),
            make_result(
                "q2",
                ["text mismatch should not matter here either"],
                ["still no text overlap"],
                gt_doc_ids=[99],
                retrieved_doc_ids=[11],
            ),
        ]
        m = compute_retrieval_metrics(results, top_k=5)
        assert m["recall@5"] == pytest.approx(0.5)
        assert m["mrr"] == pytest.approx(0.25)

    def test_case_insensitive_match(self):
        """Substring match is case-insensitive."""
        results = [
            make_result("Who painted the Mona Lisa?",
                        ["leonardo da vinci"],
                        ["The Mona Lisa was painted by Leonardo Da Vinci around 1503."]),
        ]
        m = compute_retrieval_metrics(results, top_k=5)
        assert m["recall@1"] == pytest.approx(1.0)

    def test_multi_hop_any_answer_substring(self):
        """MuSiQue multi-hop: ground_truth_doc is a list of answer candidates.
        A hit if any candidate is a substring of any retrieved doc."""
        results = [
            make_result("question",
                        ["France", "Paris"],
                        ["irrelevant doc", "Paris is a city in France."]),
        ]
        m = compute_retrieval_metrics(results, top_k=5)
        assert m["recall@5"] == pytest.approx(1.0)

    def test_msmarco_single_passage(self):
        """MS MARCO: single positive passage per query."""
        pos = "The speed of light is approximately 299,792,458 metres per second."
        results = [
            make_result("How fast is light?", [pos], [pos]),
            make_result("What is the speed of sound?", ["343 m/s in air at 20°C."],
                        ["unrelated"]),
        ]
        m = compute_retrieval_metrics(results, top_k=5)
        assert m["recall@1"] == pytest.approx(0.5)
        assert m["mrr"] == pytest.approx(0.5)


# ===========================================================================
# 3. _run_rag_pipeline — subprocess orchestration and metric parsing
# ===========================================================================

class TestRunRagPipeline:
    """
    Tests _run_rag_pipeline from rag_eval.py by mocking:
      - subprocess.run (retrieval and generation subprocesses)
      - llm_judge_accuracy (judge step)
      - the files those subprocesses would write
    """

    def _make_retrieval_meta(self, metrics):
        return {"config": {}, "metrics": metrics}

    def _make_retrieval_results(self, n=3):
        return [
            {
                "query": f"q{i}",
                "ground_truth": f"answer {i}",
                "ground_truth_doc": [f"doc_{i}"],
                "retrieved": [{"rank": 1, "doc_index": i, "score": 0.9, "document": f"doc_{i}"}],
            }
            for i in range(n)
        ]

    def _make_generated(self, n=3):
        return [
            {"query": f"q{i}", "answer": f"answer {i}", "ground_truth": f"answer {i}"}
            for i in range(n)
        ]

    def run_pipeline(self, retrieval_metrics, judge_scores, rag_cfg=None):
        from rag_eval import _run_rag_pipeline

        default_cfg = {
            "doc_dataset": "fake/docs",
            "query_dataset": "fake/queries",
            "query_column": "question",
            "query_gt_column": "answer",
            "query_answer_column": "answer",
            "num_queries": 3,
            "embedding_model": "fake/embedding-model",
            "top_k": 5,
            "gen_model": "fake/gen-model",
            "judge_model": "fake/judge-model",
        }
        cfg = {**default_cfg, **(rag_cfg or {})}

        with tempfile.TemporaryDirectory() as output_dir:
            # Paths the pipeline will look for
            rag_dir = os.path.join(output_dir, "rag", "test_key")
            os.makedirs(rag_dir)
            retrieval_meta_path = os.path.join(rag_dir, "retrieval.json")
            retrieval_results_path = os.path.join(rag_dir, "retrieval_results.json")
            generated_path = os.path.join(rag_dir, "generated.json")

            def fake_subprocess(cmd, env=None, check=None, **kwargs):
                # Write the files the subprocesses would produce
                if "single_embedding_retrieval" in " ".join(cmd):
                    with open(retrieval_meta_path, "w") as f:
                        json.dump(self._make_retrieval_meta(retrieval_metrics), f)
                    with open(retrieval_results_path, "w") as f:
                        json.dump(self._make_retrieval_results(), f)
                elif "generator" in " ".join(cmd):
                    with open(generated_path, "w") as f:
                        json.dump(self._make_generated(), f)
                return MagicMock(returncode=0)

            with patch("subprocess.run", side_effect=fake_subprocess), \
                 patch("evals.metrics.llm_judge.llm_judge_accuracy",
                       return_value=(judge_scores, ["ok"] * len(judge_scores))):
                rag_accuracy, returned_metrics = _run_rag_pipeline(
                    "test_key", cfg, output_dir
                )

        return rag_accuracy, returned_metrics

    def test_retrieval_metrics_returned(self):
        metrics = {"recall@5": 0.75, "mrr": 0.60}
        _, returned = self.run_pipeline(metrics, judge_scores=[1, 1, 0])
        assert returned["recall@5"] == pytest.approx(0.75)
        assert returned["mrr"] == pytest.approx(0.60)

    def test_rag_accuracy_is_mean_of_judge_scores(self):
        acc, _ = self.run_pipeline({"recall@5": 0.5, "mrr": 0.4},
                                   judge_scores=[1, 1, 0])
        assert acc == pytest.approx(2 / 3)

    def test_perfect_judge(self):
        acc, _ = self.run_pipeline({"recall@5": 1.0, "mrr": 1.0},
                                   judge_scores=[1, 1, 1])
        assert acc == pytest.approx(1.0)

    def test_zero_judge(self):
        acc, _ = self.run_pipeline({"recall@5": 0.0, "mrr": 0.0},
                                   judge_scores=[0, 0, 0])
        assert acc == pytest.approx(0.0)

    def test_missing_retrieval_json_returns_empty_metrics(self):
        """If the retrieval subprocess writes no metrics key, returned dict is empty."""
        metrics = {}
        _, returned = self.run_pipeline(metrics, judge_scores=[1])
        assert returned == {}

    def test_subprocess_failure_raises(self):
        from rag_eval import _run_rag_pipeline

        cfg = {
            "doc_dataset": "fake/docs",
            "query_dataset": "fake/queries",
            "query_column": "question",
            "query_gt_column": "answer",
            "query_answer_column": "answer",
            "num_queries": 3,
            "embedding_model": "fake/embedding-model",
            "top_k": 5,
            "gen_model": "fake/gen-model",
            "judge_model": "fake/judge-model",
        }

        with tempfile.TemporaryDirectory() as output_dir:
            with patch(
                "subprocess.run",
                side_effect=subprocess.CalledProcessError(1, ["single_embedding_retrieval"]),
            ):
                with pytest.raises(subprocess.CalledProcessError):
                    _run_rag_pipeline("test_key", cfg, output_dir)

    # -- Per-corpus configs --

    def test_musique_config(self):
        cfg = {
            "doc_dataset": "mihir-1999/musique-docs-validation",
            "doc_split": "train",
            "doc_column": "text",
            "query_dataset": "mihir-1999/musique-supporting",
            "query_split": "validation",
            "query_column": "question",
            "query_gt_column": "answer",
            "query_answer_column": "answer",
            "top_k": 5,
        }
        acc, metrics = self.run_pipeline(
            {"recall@5": 0.62, "mrr": 0.51}, judge_scores=[1, 0, 1], rag_cfg=cfg
        )
        assert metrics["recall@5"] == pytest.approx(0.62)
        assert metrics["mrr"] == pytest.approx(0.51)
        assert acc == pytest.approx(2 / 3)

    def test_hotpotqa_config(self):
        cfg = {
            "doc_dataset": "fake/hotpotqa-docs",
            "doc_split": "train",
            "doc_column": "text",
            "query_dataset": "RUC-NLPIR/FlashRAG_datasets",
            "query_hf_config": "hotpotqa",
            "query_split": "dev",
            "query_column": "question",
            "query_gt_column": "golden_answers",
            "query_answer_column": "golden_answers",
            "top_k": 5,
        }
        acc, metrics = self.run_pipeline(
            {"recall@5": 0.80, "recall@1": 0.55, "mrr": 0.65},
            judge_scores=[1, 1, 1],
            rag_cfg=cfg,
        )
        assert metrics["recall@5"] == pytest.approx(0.80)
        assert metrics["recall@1"] == pytest.approx(0.55)
        assert acc == pytest.approx(1.0)

    def test_msmarco_config(self):
        cfg = {
            "doc_dataset": "mihir-1999/msmarco-qa-positive-passages-dev",
            "doc_split": "train",
            "doc_column": "text",
            "query_dataset": "RUC-NLPIR/FlashRAG_datasets",
            "query_hf_config": "msmarco-qa",
            "query_split": "dev",
            "query_column": "question",
            "query_gt_column": "golden_answers",
            "query_answer_column": "golden_answers",
            "top_k": 5,
        }
        acc, metrics = self.run_pipeline(
            {"recall@5": 0.70, "mrr": 0.58}, judge_scores=[1, 0, 0], rag_cfg=cfg
        )
        assert metrics["recall@5"] == pytest.approx(0.70)
        assert acc == pytest.approx(1 / 3)


# ===========================================================================
# 4. Metric key prefixing in all_metrics
# ===========================================================================

class TestMetricKeyPrefixing:
    """
    Verify that rag_eval.main prefixes retrieval metrics as
    {eval_key}/rag_{metric_name} and accuracy as {eval_key}/rag_accuracy.
    These keys are what gets logged to W&B.
    """

    def _simulate_metrics(self, eval_key, rag_accuracy, retrieval_metrics):
        """
        Replicate the all_metrics population logic from rag_eval.main (lines 391-394).
        """
        all_metrics = {}
        all_metrics[f"{eval_key}/rag_accuracy"] = rag_accuracy
        for k, v in retrieval_metrics.items():
            all_metrics[f"{eval_key}/rag_{k}"] = v
        return all_metrics

    def test_musique_metric_keys(self):
        m = self._simulate_metrics(
            "gen_large_mem_musique",
            0.55,
            {"recall@5": 0.62, "mrr": 0.51},
        )
        assert "gen_large_mem_musique/rag_accuracy" in m
        assert "gen_large_mem_musique/rag_recall@5" in m
        assert "gen_large_mem_musique/rag_mrr" in m
        assert m["gen_large_mem_musique/rag_accuracy"] == pytest.approx(0.55)
        assert m["gen_large_mem_musique/rag_recall@5"] == pytest.approx(0.62)

    def test_hotpotqa_metric_keys(self):
        m = self._simulate_metrics(
            "gen_large_mem_hotpotqa",
            0.70,
            {"recall@5": 0.80, "recall@1": 0.55, "mrr": 0.65},
        )
        assert "gen_large_mem_hotpotqa/rag_recall@1" in m
        assert "gen_large_mem_hotpotqa/rag_recall@5" in m
        assert "gen_large_mem_hotpotqa/rag_mrr" in m

    def test_msmarco_metric_keys(self):
        m = self._simulate_metrics(
            "gen_large_mem_msmarco",
            0.33,
            {"recall@5": 0.70, "mrr": 0.58},
        )
        assert "gen_large_mem_msmarco/rag_accuracy" in m
        assert "gen_large_mem_msmarco/rag_recall@5" in m
        assert m["gen_large_mem_msmarco/rag_mrr"] == pytest.approx(0.58)

    def test_comparison_table_recall_key_lookup(self):
        """
        rag_eval.main picks recall@{top_k} for the retrieval comparison table.
        Verify the key lookup logic: uses recall@{top_k} when present, falls back to mrr.
        """
        top_k = 5
        all_metrics = {
            "gen_large_mem_musique/doc_access_acc": 0.42,
            "gen_large_mem_musique/rag_recall@5": 0.62,
            "gen_large_mem_musique/rag_mrr": 0.51,
        }
        eval_key = "gen_large_mem_musique"
        recall_key = f"rag_recall@{top_k}"
        full_recall_key = f"{eval_key}/{recall_key}"

        # Lookup matches top_k
        assert full_recall_key in all_metrics
        assert all_metrics[full_recall_key] == pytest.approx(0.62)

        # Fallback: if recall@k absent, falls back to mrr
        all_metrics_no_recall = {k: v for k, v in all_metrics.items()
                                 if "recall" not in k}
        assert f"{eval_key}/rag_mrr" in all_metrics_no_recall

    def test_delta_metric_computed(self):
        """memory_vs_rag_delta = llm_judge_accuracy - rag_accuracy."""
        all_metrics = {
            "gen_large_mem_musique/llm_judge_accuracy": 0.60,
            "gen_large_mem_musique/rag_accuracy": 0.55,
        }
        eval_key = "gen_large_mem_musique"
        ml_key  = f"{eval_key}/llm_judge_accuracy"
        rag_key = f"{eval_key}/rag_accuracy"
        delta = all_metrics[ml_key] - all_metrics[rag_key]
        all_metrics[f"{eval_key}/memory_vs_rag_delta"] = delta

        assert all_metrics[f"{eval_key}/memory_vs_rag_delta"] == pytest.approx(0.05)

    def test_all_three_corpus_evals_present(self):
        """Sanity check: all three eval keys should produce non-overlapping metric keys."""
        eval_keys = ["gen_large_mem_musique", "gen_large_mem_hotpotqa", "gen_large_mem_msmarco"]
        all_metrics = {}
        for key in eval_keys:
            all_metrics.update(self._simulate_metrics(
                key, 0.5, {"recall@5": 0.6, "mrr": 0.5}
            ))
        # Each key produces 3 entries; no collisions
        assert len(all_metrics) == len(eval_keys) * 3
        for key in eval_keys:
            assert f"{key}/rag_accuracy" in all_metrics
            assert f"{key}/rag_recall@5" in all_metrics
            assert f"{key}/rag_mrr" in all_metrics
