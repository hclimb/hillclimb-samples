# Benchmark-reference loss vs benchmark accuracy

| setting | method | test loss | accuracy |
|---|---|---:|---:|
| reason_math | FullTraining | 0.6174 | 56.60 |
| reason_math | LayerwiseRaw | 0.6072 | 48.60 |
| reason_math | LayerwiseSoft | 0.6059 | 53.00 |
| reason_math | LayerwiseSoftP | 0.6052 | 39.40 |
| reason_math | LayerwiseOptA | 0.6064 | 51.60 |
| reason_math | TargetOnly | 0.7202 | - |
| reason_code | FullTraining | 0.5367 | 53.97 |
| reason_code | LayerwiseRaw | 0.5542 | 37.04 |
| reason_code | LayerwiseSoft | 0.5568 | 53.97 |
| reason_code | LayerwiseSoftP | 0.6752 | 33.60 |
| reason_code | LayerwiseOptA | 0.5454 | 35.71 |
| reason_code | TargetOnly | 0.9234 | - |
| mixed_math | FullTraining | 0.6179 | 56.80 |
| mixed_math | LayerwiseRaw | 0.6026 | 13.40 |
| mixed_math | LayerwiseSoft | 0.6021 | 29.60 |
| mixed_math | LayerwiseSoftP | 0.6031 | 0.00 |
| mixed_math | LayerwiseOptA | 0.6023 | 32.20 |
