from models import qwen3, qwen3_mem, qwen3_mem_embed, qwen3_distill, qwen3_msa
from .output import ModelOutput

def get_model(model_cfg, tp_devices):
    if model_cfg.name == "qwen3":
        return qwen3.init(model_cfg, tp_devices)
    elif model_cfg.name == "qwen3_msa":
        return qwen3_msa.init(model_cfg, tp_devices)
    elif model_cfg.name == "qwen3_mem":
        return qwen3_mem.init(model_cfg, tp_devices)
    elif model_cfg.name == "qwen3_mem_embed":
        return qwen3_mem_embed.init(model_cfg, tp_devices)
    elif model_cfg.name == "qwen3_distill":
        return qwen3_distill.init(model_cfg, tp_devices)
    else:
        raise ValueError(f"Unknown model name: {model_cfg.name}")


__all__ = ["ModelOutput", "get_model"]
