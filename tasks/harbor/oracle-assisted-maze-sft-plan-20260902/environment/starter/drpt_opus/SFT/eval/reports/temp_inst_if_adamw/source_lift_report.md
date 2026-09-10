# Source-level selection lift, all five settings

`lift` = P(source | selected) / P(source | candidate). 1.00 means no preference.

`len` is the mean assistant-response length in characters. `D* len` on each
section header is the same statistic for that setting's target split, so a
source's distance from it is readable at a glance.


## inst_if  (target `precise_if`, D* len 1774 chars)

| source | domain | len | Raw | OptA | Soft | SoftP |
|---|---|---:|---:|---:|---:|---:|
| Dolci Instruct Precise IF **←target domain** | `precise_if` | 1270 | 1.153 | 1.156 | 1.126 | 1.505 |
| Logic Puzzles | `other` | 225 | 1.056 | 1.034 | 1.076 | 1.495 |
| FLAN | `other` | 102 | 1.020 | 1.040 | 1.069 | 2.002 |
| Aya | `multilingual` | 448 | 0.975 | 0.967 | 1.003 | 1.617 |
| CoCoNot | `safety` | 1161 | 1.010 | 1.023 | 1.000 | 0.727 |
| Verifiable Reasoning | `reasoning` | 509 | 1.016 | 1.018 | 0.998 | 0.724 |
| TableGPT | `other` | 212 | 1.004 | 1.004 | 0.989 | 0.377 |
| WildJailbreak | `safety` | 1341 | 0.989 | 0.989 | 0.977 | 0.598 |
| WildGuardMix | `safety` | 453 | 0.922 | 0.916 | 0.970 | 0.893 |
| Wildchat | `chat` | 2547 | 0.974 | 0.988 | 0.926 | 0.492 |
| Hardcoded Data | `hardcoded_data` | 463 | 0.886 | 0.874 | 0.893 | 0.247 |
| OpenAssistant | `chat` | 1245 | 0.893 | 0.880 | 0.872 | 0.575 |

## reason_math  (target `math`, D* len 521 chars)

| source | domain | len | Raw | OptA | Soft | SoftP |
|---|---|---:|---:|---:|---:|---:|
| SciRiff | `science` | 359 | 1.283 | 1.294 | 1.272 | 2.480 |
| Tulu 3 Persona Python | `coding` | 583 | 1.162 | 1.193 | 1.171 | 0.755 |
| Logic Puzzles | `other` | 305 | 1.243 | 1.243 | 1.122 | 3.111 |
| Dolci Instruct Python Algorithms | `coding` | 845 | 1.079 | 1.122 | 1.048 | 0.672 |
| Tulu 3 Persona MATH **←target domain** | `math` | 2777 | 0.933 | 0.916 | 1.017 | 0.339 |
| Tulu 3 Persona Algebra **←target domain** | `math` | 2183 | 0.929 | 0.885 | 1.015 | 0.327 |
| Verifiable Reasoning | `reasoning` | 515 | 0.943 | 0.934 | 0.956 | 1.739 |
| Evol CodeAlpaca | `coding` | 1550 | 0.943 | 0.928 | 0.910 | 0.753 |
| OpenMathInstruct 2 **←target domain** | `math` | 504 | 0.858 | 0.888 | 0.861 | 1.161 |
| Tulu 3 Persona GSM **←target domain** | `math` | 1241 | 0.840 | 0.824 | 0.826 | 0.596 |
| Dolci Instruct OpenThoughts3+ Science | `science` | 2729 | 0.812 | 0.755 | 0.820 | 0.519 |

## reason_code  (target `mbpp`, D* len 199 chars)

| source | domain | len | Raw | OptA | Soft | SoftP |
|---|---|---:|---:|---:|---:|---:|
| SciRiff | `science` | 359 | 1.177 | 1.168 | 1.165 | 2.342 |
| Tulu 3 Persona Algebra | `math` | 2183 | 1.031 | 1.079 | 1.098 | 0.225 |
| Logic Puzzles | `other` | 305 | 1.084 | 1.131 | 1.097 | 3.973 |
| OpenMathInstruct 2 | `math` | 504 | 1.080 | 1.071 | 1.065 | 0.795 |
| Tulu 3 Persona MATH | `math` | 2777 | 0.986 | 1.057 | 1.063 | 0.228 |
| Tulu 3 Persona GSM | `math` | 1241 | 1.049 | 1.073 | 1.038 | 0.550 |
| Dolci Instruct OpenThoughts3+ Science | `science` | 2729 | 1.001 | 0.941 | 1.007 | 0.583 |
| Tulu 3 Persona Python | `coding` | 583 | 1.034 | 1.006 | 0.962 | 0.841 |
| Verifiable Reasoning | `reasoning` | 515 | 0.912 | 0.882 | 0.910 | 1.101 |
| Evol CodeAlpaca | `coding` | 1550 | 0.909 | 0.894 | 0.905 | 0.788 |
| Dolci Instruct Python Algorithms | `coding` | 845 | 0.874 | 0.844 | 0.861 | 0.926 |

## mixed_if  (target `precise_if`, D* len 1774 chars)

| source | domain | len | Raw | OptA | Soft | SoftP |
|---|---|---:|---:|---:|---:|---:|
| Dolci Instruct Precise IF **←target domain** | `precise_if` | 1266 | 1.249 | 1.266 | 1.213 | 1.944 |
| FLAN | `other` | 97 | 1.069 | 1.079 | 1.131 | 2.972 |
| Dolci Instruct OpenThoughts3+ Science | `science` | 2697 | 1.225 | 1.241 | 1.123 | 0.643 |
| Logic Puzzles | `other` | 286 | 1.067 | 1.038 | 1.107 | 1.794 |
| CoCoNot | `safety` | 1164 | 1.050 | 1.081 | 1.065 | 1.025 |
| Verifiable Reasoning | `reasoning` | 512 | 1.019 | 1.019 | 1.052 | 1.043 |
| WildGuardMix | `safety` | 459 | 0.962 | 0.964 | 1.037 | 1.397 |
| Hardcoded Data | `hardcoded_data` | 447 | 1.009 | 1.014 | 1.022 | 1.906 |
| SciRiff | `science` | 352 | 0.966 | 0.966 | 1.022 | 1.352 |
| Aya | `multilingual` | 484 | 0.992 | 0.960 | 1.009 | 1.862 |
| Dolci Instruct Python Algorithms | `coding` | 859 | 1.077 | 1.060 | 0.997 | 0.402 |
| TableGPT | `other` | 210 | 0.974 | 0.974 | 0.996 | 0.655 |
| OpenAssistant | `chat` | 1260 | 0.931 | 0.949 | 0.988 | 1.247 |
| WildJailbreak | `safety` | 1342 | 0.941 | 0.965 | 0.983 | 0.879 |
| Wildchat | `chat` | 2430 | 0.986 | 1.001 | 0.979 | 0.953 |
| OpenMathInstruct 2 | `math` | 502 | 0.917 | 0.946 | 0.971 | 0.469 |
| Tulu 3 Persona Python | `coding` | 576 | 0.951 | 0.918 | 0.930 | 0.291 |
| Evol CodeAlpaca | `coding` | 1548 | 0.915 | 0.935 | 0.909 | 0.614 |
| Tulu 3 Persona GSM | `math` | 1194 | 0.950 | 0.981 | 0.895 | 0.382 |
| Tulu 3 Persona Algebra | `math` | 2186 | 0.971 | 0.951 | 0.875 | 0.105 |
| Tulu 3 Persona MATH | `math` | 2754 | 0.829 | 0.793 | 0.742 | 0.102 |

## mixed_math  (target `math`, D* len 521 chars)

| source | domain | len | Raw | OptA | Soft | SoftP |
|---|---|---:|---:|---:|---:|---:|
| SciRiff | `science` | 352 | 1.279 | 1.314 | 1.309 | 2.081 |
| TableGPT | `other` | 210 | 1.215 | 1.280 | 1.256 | 0.905 |
| Tulu 3 Persona Python | `coding` | 576 | 1.210 | 1.276 | 1.216 | 0.671 |
| FLAN | `other` | 97 | 1.185 | 1.197 | 1.175 | 3.043 |
| Aya | `multilingual` | 484 | 1.117 | 1.129 | 1.134 | 1.927 |
| Logic Puzzles | `other` | 286 | 1.136 | 1.121 | 1.132 | 1.968 |
| Dolci Instruct Python Algorithms | `coding` | 859 | 1.052 | 1.085 | 1.036 | 0.339 |
| Hardcoded Data | `hardcoded_data` | 447 | 1.114 | 1.085 | 1.007 | 0.763 |
| WildGuardMix | `safety` | 459 | 0.982 | 1.002 | 0.991 | 1.189 |
| OpenAssistant | `chat` | 1260 | 1.008 | 0.976 | 0.991 | 1.162 |
| Verifiable Reasoning | `reasoning` | 512 | 0.934 | 0.964 | 0.984 | 1.199 |
| Evol CodeAlpaca | `coding` | 1548 | 0.945 | 0.947 | 0.932 | 0.469 |
| Wildchat | `chat` | 2430 | 0.918 | 0.903 | 0.928 | 0.568 |
| OpenMathInstruct 2 **←target domain** | `math` | 502 | 0.949 | 0.950 | 0.927 | 0.817 |
| Tulu 3 Persona MATH **←target domain** | `math` | 2754 | 0.919 | 0.868 | 0.916 | 0.209 |
| Dolci Instruct Precise IF | `precise_if` | 1266 | 0.889 | 0.892 | 0.882 | 1.047 |
| Tulu 3 Persona Algebra **←target domain** | `math` | 2186 | 0.854 | 0.806 | 0.847 | 0.207 |
| Tulu 3 Persona GSM **←target domain** | `math` | 1194 | 0.879 | 0.851 | 0.829 | 0.479 |
| CoCoNot | `safety` | 1164 | 0.866 | 0.824 | 0.820 | 0.672 |
| WildJailbreak | `safety` | 1342 | 0.811 | 0.778 | 0.813 | 0.873 |
| Dolci Instruct OpenThoughts3+ Science | `science` | 2697 | 0.803 | 0.734 | 0.801 | 0.325 |

## Does response length explain the ordering?

Pearson r between a source's |len - D* len| and its lift, pooled over settings.
A strong negative r would mean curation is largely tracking response length
rather than content.

| method | n | r |
|---|---:|---:|
| LayerwiseRaw | 76 | -0.373 |
| LayerwiseOptA | 76 | -0.408 |
| LayerwiseSoft | 76 | -0.223 |
| LayerwiseSoftP | 76 | -0.319 |

![source lift](source_lift.png)

