from .llm_judge import llm_judge_accuracy, llm_judge_score
from .lexical_grounding import lexical_grounding

METRICS = {
    "llm_judge_accuracy": llm_judge_accuracy,
    "llm_judge_score": llm_judge_score,
    "lexical_grounding": lexical_grounding,
}


def run_metrics(results, metrics_cfg):
    """
    Run all configured metrics over results.

    Args:
        results: list of dicts with at least {"generated", "ground_truth"} keys.
        metrics_cfg: OmegaConf/dict mapping metric_name -> metric kwargs.
        model: loaded model object (used for tokenizer + forward).

    Returns:
        annotated_results: same as results but with per-sample metric scores added.
        aggregate_metrics: {metric_name: mean_score}.
    """
    if not metrics_cfg:
        return results, {}

    annotated = [dict(r) for r in results]
    aggregate = {}

    for metric_name, metric_kwargs in metrics_cfg.items():
        if metric_name not in METRICS:
            raise ValueError(
                f"Unknown metric '{metric_name}'. Available: {list(METRICS)}"
            )
        fn = METRICS[metric_name]
        kwargs = dict(metric_kwargs) if metric_kwargs else {}

        result = fn(results, **kwargs)
        scores, outputs = result if isinstance(result, tuple) else (result, None)

        for i, score in enumerate(scores):
            annotated[i][metric_name] = score
            if outputs is not None:
                annotated[i][f"{metric_name}_output"] = outputs[i]

        # mean over defined (non-None) scores, so metrics may return None for
        # samples where the score is undefined (e.g. no content words).
        valid = [s for s in scores if s is not None]
        aggregate[metric_name] = float(sum(valid) / len(valid)) if valid else 0.0

    return annotated, aggregate
