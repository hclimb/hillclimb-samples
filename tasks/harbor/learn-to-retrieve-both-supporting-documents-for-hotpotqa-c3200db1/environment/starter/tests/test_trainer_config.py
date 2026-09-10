"""Quick test: verify trainer/standard.yaml composes with expected structure."""
import os
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("add", lambda x, y: x + y)
OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)

config_dir = os.path.abspath("configs")

with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
    # Load just the trainer config group
    cfg = compose(config_name="train", overrides=["trainer=standard", "model=qwen3", "dataset=squad"])

trainer = cfg.trainer

# Top-level trainer fields
assert trainer.steps == 100000
assert trainer.learning_rate == 1e-4
assert trainer.eval_interval == 5000

# All three evals present
assert set(trainer.evals.keys()) == {"nll_squad", "nll_bior", "generation_embed_squad"}, \
    f"Got keys: {set(trainer.evals.keys())}"

# nll_squad: eval type from eval/nll.yaml, dataset from dataset/squad.yaml
nll_sq = trainer.evals.nll_squad
assert nll_sq.type == "nll",            f"type: {nll_sq.type}"
assert nll_sq.dataset.name == "squad",  f"dataset.name: {nll_sq.dataset.name}"
assert nll_sq.dataset.split == "val",   f"split: {nll_sq.dataset.split}"
assert nll_sq.dataset.qa_limit == 800,  f"qa_limit: {nll_sq.dataset.qa_limit}"
assert nll_sq.dataset.limit == 4000,    f"limit: {nll_sq.dataset.limit}"  # 1*(0+5*800)

# nll_bior: dataset from dataset/bior.yaml
nll_bi = trainer.evals.nll_bior
assert nll_bi.type == "nll"
assert nll_bi.dataset.name == "bior"
assert nll_bi.dataset.limit == 4800,    f"limit: {nll_bi.dataset.limit}"  # 1*(0+6*800)

# generation_embed_squad: eval type from eval/generation_embed.yaml
gen = trainer.evals.generation_embed_squad
assert gen.type == "generation_embed"
assert gen.max_new_tokens == 16
assert gen.num_samples == 32
assert gen.dataset.name == "squad"
assert gen.dataset.split == "val"

# aux_losses untouched
assert trainer.aux_losses.doc_access_loss.enabled == True
assert trainer.aux_losses.doc_access_loss.weight == 0.1
assert trainer.aux_losses.mem_uniform_kl.enabled == False

print("All assertions passed!\n")
print("=== trainer config ===")
print(OmegaConf.to_yaml(trainer))
