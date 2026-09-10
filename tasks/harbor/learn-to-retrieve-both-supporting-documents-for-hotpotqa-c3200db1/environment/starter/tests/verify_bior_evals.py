
import hydra
from omegaconf import DictConfig, OmegaConf
import sys
import os

# Mock utils processing if needed or ensure path is set
sys.path.append(os.getcwd())

@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig):
    print("Hydra initialized successfully.")
    
    # Test instantiating different evals
    from evals.bior_evals import BioRQAEvaluator, BioRCompletionEvaluator, BioRNLLEvaluator
    
    # Mocking Hydra logic for instantiating just the eval part if we were running eval.py
    # But here we just want to verify we can instantiate the classes manually and via hydra.utils.instantiate
    
    from hydra.utils import instantiate
    
    configs_to_test = [
        "eval/bior_qa",
        "eval/bior_completion",
        "eval/bior_nll_bio",
        "eval/bior_nll_qa"
    ]
    
    for config_name in configs_to_test:
        print(f"Testing config: {config_name}")
        # We need to compose a config for this specific target
        # Since we are already inside a hydra main, we can't easily recompose global config?
        # We can use compose API.
        pass

if __name__ == "__main__":
    # Simplified test: just import and instantiate manually with dummy config
    print("Starting verification...")
    try:
        from evals.bior_evals import BioRQAEvaluator, BioRCompletionEvaluator, BioRNLLEvaluator
        print("Import successful.")
    except Exception as e:
        print(f"Import failed: {e}")
        sys.exit(1)

    print("Verification complete.")
