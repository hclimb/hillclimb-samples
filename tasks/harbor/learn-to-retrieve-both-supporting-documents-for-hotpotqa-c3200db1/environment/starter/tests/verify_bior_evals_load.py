
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
import sys
import os

# Ensure project root is in path
sys.path.append(os.getcwd())

def test_configs():
    print("Testing Configs with OmegaConf.load...")
    
    config_dir = "configs/eval"
    files = [
        "bior_qa.yaml", 
        "bior_completion.yaml", 
        "bior_nll_bio.yaml", 
        "bior_nll_qa.yaml"
    ]
    
    for fname in files:
        fpath = os.path.join(config_dir, fname)
        print(f"Loading {fpath}...")
        try:
            cfg = OmegaConf.load(fpath)
            # print(OmegaConf.to_yaml(cfg))
            
            target = cfg.get("_target_")
            print(f"  _target_: {target}")
            
            if not target:
                print("  ERROR: _target_ missing!")
                continue
                
            # Manual instantiation check
            from hydra.utils import get_class
            try:
                Class = get_class(target)
                print(f"  Class resolved: {Class.__name__}")
                
                # Mock instantiation: Class(cfg)
                # Note: BioR classes might need 'dataset' argument if they were truly instantiated via eval.py loop?
                # No, they are instantiated as Evaluator(cfg). 
                # The evaluate(model, dataset) method is called LATER.
                
                obj = Class(cfg)
                print(f"  Instantiation success: {type(obj)}")
                
            except Exception as e:
                print(f"  Instantiation failed: {e}")
                
        except Exception as e:
            print(f"  Load failed: {e}")

if __name__ == "__main__":
    test_configs()
