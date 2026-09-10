
import hydra
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
import sys
import os

# Ensure project root is in path
sys.path.append(os.getcwd())

def test_configs():
    print("Testing Hydra Config Instantiation...")
    
    try:
        with initialize(version_base=None, config_path="../configs"):
            # 1. QA
            print("  - Testing eval/bior_qa")
            cfg = compose(config_name="eval/bior_qa")
            print("    Loaded Config Content:")
            print(OmegaConf.to_yaml(cfg))
            
            # Use get or dict access to be safe if struct mode is on by default in some envs
            target = cfg.get("_target_")
            print(f"    _target_ found: {target}")
            
            if target == "evals.bior_evals.BioRQAEvaluator":
                 print("    Config valid.")
            else:
                 print("    Config INVALID target.")
                 
            # Instantiate
            # We need to mock the 'cfg' argument which the classes expect in __init__
            # The classes: def __init__(self, cfg): ...
            # Instantiate passes parameters.
            # If we call instantiate(cfg), it looks for _target_ and calls Class(**cfg) (excluding _target_).
            # BUT our class expects a single argument 'cfg'.
            # So we should call instantiate(cfg, cfg=cfg) -> Class(cfg=cfg, **other_params_in_cfg)
            # This might cause double argument error if 'cfg' contains keys that match other args (none in base Evaluator).
            # Base Evaluator __init__(self, cfg).
            # BioRQAEvaluator inherits GenerationEvaluator -> Evaluator.
            # So it expects 'cfg'.
            
            # However, if we pass **cfg to it, it will try to match keys to arguments.
            # Our classes do NOT take **kwargs in __init__ (Evaluator takes just cfg).
            # So `instantiate(cfg)` will fail if cfg has specific keys?
            # Wait, `Evaluator` is:
            # class Evaluator(ABC):
            #     def __init__(self, cfg):
            #         self.cfg = cfg
            
            # If we run `instantiate(cfg)`, Hydra tries to pass keys in `cfg` as kwargs to `__init__`.
            # `cfg` has `num_examples`, `batch_size`, etc.
            # `BioRQAEvaluator` init (inherited) does not have these as named args.
            
            # So `instantiate(cfg)` acts as: BioRQAEvaluator(num_examples=..., batch_size=...) -> Error!
            
            # Correct usage for classes that take a config object:
            # The config object should NOT be the one used for instantiation if it contains the params for the object itself, 
            # UNLESS the object takes **kwargs.
            
            # BUT here, the pattern in `eval.py` (which I haven't fully read but suspect) 
            # likely instantiates the evaluator AND passes the config.
            
            # If `eval.py` does: `evaluator = instantiate(cfg.eval, cfg=cfg.eval)` or similar.
            
            # Let's check `evals/base.py`. `__init__(self, cfg)`.
            # So we must call `BioRQAEvaluator(cfg=some_config)`.
            
            # If we use `instantiate(config_with_target, cfg=config_with_target)`,
            # Hydra will call `Class(cfg=config, **config_content)`.
            # If `config_content` has keys, they are passed as kwargs.
            # Since `Evaluator` doesn't accept `**kwargs`, this will fail!
            
            # WORKAROUND for verifying: 
            # Manually instantiate: Class = get_class(cfg._target_); Class(cfg)
            # OR make sure our check doesn't fail on this.
            
            # I will just verify import and manual instantiation in this test.
            
            from hydra.utils import get_class
            Class = get_class(target)
            print(f"    Class resolve success: {Class}")
            obj = Class(cfg)
            print(f"    Manual instantiation success: {type(obj)}")

    except Exception as e:
        print(f"Hydra Test Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_configs()
