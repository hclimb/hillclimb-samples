# Two-Pass Memory Training: Scale Comparison

Checkpoints: **256k mem** (step 101000) vs **128k mem** (step 102250)
Eval: gen_embed, 32k–256k memory tokens, 256 samples each.

| ![256k mem](mem_scale_sweep_256k_mem.png) | ![128k mem](mem_scale_sweep_128k_mem.png) |
|:---:|:---:|
| 256k mem | 128k mem |

## MS MARCO QA

| Mem tokens | 256k mem judge acc | 256k mem retrieval acc | 128k mem judge acc | 128k mem retrieval acc |
|:----------:|:-------------------:|:----------------------:|:-------------------:|:----------------------:|
|        32k |               30.5% |                   8.0% |               28.9% |                   7.7% |
|       256k |               27.3% |                   5.0% |               29.3% |                   4.8% |
|       512k |               30.9% |                    N/A |               27.3% |                    N/A |
|         1M |               24.6% |                    N/A |               27.7% |                    N/A |

## HotpotQA

| Mem tokens | 256k mem judge acc | 256k mem retrieval acc | 128k mem judge acc | 128k mem retrieval acc |
|:----------:|:-------------------:|:----------------------:|:-------------------:|:----------------------:|
|        32k |               35.2% |                  64.5% |               32.0% |                  63.9% |
|       256k |               33.6% |                  36.8% |               33.6% |                  35.7% |
|       512k |               31.6% |                  28.6% |               28.5% |                  27.5% |
|         1M |               28.5% |                    N/A |               29.7% |                    N/A |

