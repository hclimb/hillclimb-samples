from .nll import NLLEvaluator
from .generation import GenerationEvaluator
from .generation_embed import GenerationEmbedEvaluator
from .gen_large_mem import GenLargeMemEvaluator

def get_evaluator(eval_cfg, key=None):
    eval_type = eval_cfg.get("type", eval_cfg.get("name"))
    key = key or eval_type
    if eval_type == "nll":
        return NLLEvaluator(eval_cfg, key=key)
    elif eval_type == "generation":
        return GenerationEvaluator(eval_cfg, key=key)
    elif eval_type == "generation_embed":
        return GenerationEmbedEvaluator(eval_cfg, key=key)
    elif eval_type == "generation_large_mem":
        return GenLargeMemEvaluator(eval_cfg, key=key)
    elif eval_type == "swap_logit_delta":
        from .swap_logit_delta import SwapLogitDeltaEvaluator
        return SwapLogitDeltaEvaluator(eval_cfg, key=key)
    elif eval_type == "generation_large_mem_msa":
        from .gen_large_mem_msa import GenLargeMemMSAEvaluator
        return GenLargeMemMSAEvaluator(eval_cfg, key=key)
    elif eval_type == "generation_large_mem_rag_hybrid":
        from .gen_large_mem_rag_hybrid import GenLargeMemRagHybridEvaluator
        return GenLargeMemRagHybridEvaluator(eval_cfg, key=key)
    elif eval_type == "ruler":
        from .ruler import RULEREvaluator
        return RULEREvaluator(eval_cfg, key=key)
    else:
        raise ValueError(f"Unknown evaluator type: {eval_type}")
