import os
from abc import ABC, abstractmethod

class Evaluator(ABC):
    def __init__(self, cfg, key=None):
        self.cfg = cfg
        self.key = key or cfg.get("type", cfg.get("name", "eval"))

    @abstractmethod
    def evaluate(self, model, dataset, step=None, **kwargs):
        """
        Run the evaluation.

        Args:
            model: The loaded model object (must have .forward, .tokenizer, .weights, etc.)
            dataset: The loaded dataset object.
            step: Current training step, used to organise output files.

        Returns:
            A dictionary of metrics/results.
        """
        pass

    def _get_output_dir(self):
        if "EVAL_OUTPUT_DIR" in os.environ:
            return os.environ["EVAL_OUTPUT_DIR"]
        try:
            from hydra.core.hydra_config import HydraConfig
            return HydraConfig.get().runtime.output_dir
        except (ImportError, ValueError, AttributeError):
            return os.getcwd()

    def _get_output_path(self, step, filename):
        base = self._get_output_dir()
        if step is not None:
            path = os.path.join(base, "eval_results", f"step_{step}", self.key, filename)
        else:
            path = os.path.join(base, "eval_results", self.key, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    @staticmethod
    def _detect_model_type(model) -> str:
        cfg = model.cfg
        if "embed_model" in cfg:
            return "qwen3_mem_embed"
        if "mem_layers" in cfg:
            return "qwen3_mem"
        return "qwen3"
