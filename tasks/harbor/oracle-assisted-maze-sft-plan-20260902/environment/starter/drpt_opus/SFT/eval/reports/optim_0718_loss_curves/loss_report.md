# AdamW/Muon loss-curve comparison

Successful requested runs: **60 / 64**. Missing or incomplete: **4**.

## Reading the metrics

- **Target validation** is `val_loss` on the small target/proxy set used by the selection method. Lower final loss and lower normalized AUC are better, but this set is not independent of selection.
- **General held-out** is `eval_loss` on the larger evaluation set. It is the main loss-based generalization cross-check.
- **Train** compares the mean of the first and last 10% of logged steps; plots use a centered rolling mean. For merged-batch selection runs this is the returned train+target merged loss, while FullTraining returns train-only loss, so cross-method absolute train-loss ranks are not apples-to-apples.
- AUC integrates loss over normalized progress from 0 to 1, making it comparable across run lengths. Lower is better. Reduction is initial minus final, so positive is improvement.
- `hybrid` is displayed as **Muon (hybrid)**: Muon updates matrix parameters and AdamW updates the remaining parameters.

## Winners by setting

| Setting | Optimizer | Lowest target final | Lowest target AUC | Lowest general final | Lowest general AUC | Largest within-run train reduction |
|---|---|---|---|---|---|---|
| alpaca_samsum | AdamW | LayerwiseOptA | LayerwiseOptA | LayerwiseRaw | LayerwiseSoft | LayerwiseOptA |
| alpaca_samsum | Muon (hybrid) | LayerwiseOptA | GlobalOptA | GlobalOptA | GlobalOptA | FullTraining |
| less_squad | AdamW | LayerwiseSoft | LayerwiseOptA | LayerwiseSoft | LayerwiseSoft | FullTraining |
| less_squad | Muon (hybrid) | LayerwiseRaw | LayerwiseRaw | LayerwiseRaw | LayerwiseRaw | FullTraining |
| less_tydiqa | AdamW | LayerwiseSoft | LayerwiseSoft | LayerwiseSoft | LayerwiseSoft | FullTraining |
| less_tydiqa | Muon (hybrid) | LayerwiseRaw | LayerwiseOptA | LayerwiseRaw | LayerwiseOptA | FullTraining |
| triviaqa_nq | AdamW | LayerwiseRaw | LayerwiseRaw | LayerwiseRaw | LayerwiseRaw | LayerwiseRaw |
| triviaqa_nq | Muon (hybrid) | FullTraining | FullTraining | FullTraining | FullTraining | FullTraining |

## Win counts

Counts are across the four settings; ties count for every tied method. Raw losses are never averaged across different tasks.

| Optimizer | Method | Target final | Target AUC | General final | General AUC | Available settings |
|---|---|---:|---:|---:|---:|---:|
| AdamW | FullTraining | 0 | 0 | 0 | 0 | 4 |
| AdamW | GlobalRaw | 0 | 0 | 0 | 0 | 4 |
| AdamW | LayerwiseRaw | 1 | 1 | 2 | 1 | 4 |
| AdamW | GlobalOptA | 0 | 0 | 0 | 0 | 4 |
| AdamW | LayerwiseOptA | 1 | 2 | 0 | 0 | 4 |
| AdamW | GlobalSoft | 0 | 0 | 0 | 0 | 4 |
| AdamW | LayerwiseSoft | 2 | 1 | 2 | 3 | 4 |
| Muon (hybrid) | FullTraining | 1 | 1 | 1 | 1 | 4 |
| Muon (hybrid) | GlobalRaw | 0 | 0 | 0 | 0 | 4 |
| Muon (hybrid) | LayerwiseRaw | 2 | 1 | 2 | 1 | 4 |
| Muon (hybrid) | GlobalOptA | 0 | 1 | 1 | 1 | 4 |
| Muon (hybrid) | LayerwiseOptA | 1 | 1 | 0 | 1 | 4 |
| Muon (hybrid) | GlobalMuonSur | 0 | 0 | 0 | 0 | 4 |
| Muon (hybrid) | LayerwiseMuonSur | 0 | 0 | 0 | 0 | 4 |
| Muon (hybrid) | GlobalSoft | 0 | 0 | 0 | 0 | 0 |
| Muon (hybrid) | LayerwiseSoft | 0 | 0 | 0 | 0 | 4 |

## Detailed loss reductions

### alpaca_samsum · AdamW

| Method | Target initial → final (reduction; AUC) | General initial → final (reduction; AUC) | Train early → late (reduction; AUC) | W&B |
|---|---:|---:|---:|---|
| FullTraining | 2.2198 → 1.6084 (0.6114; 1.6110) | 2.0594 → 1.7549 (0.3045; 1.7594) | 1.4687 → 1.3771 (0.0915; 1.3926) | [run](https://wandb.ai/leena12/drpt_opus/runs/jp7cpzgr) |
| GlobalRaw | 2.2198 → 1.4081 (0.8117; 1.4407) | 2.0594 → 1.6604 (0.3990; 1.6779) | 1.4911 → 1.3941 (0.0969; 1.4099) | [run](https://wandb.ai/leena12/drpt_opus/runs/f1e69qcq) |
| LayerwiseRaw | 2.2198 → 1.0856 (1.1342; 1.1585) | 2.0594 → 1.5767 (0.4827; 1.5978) | 1.4884 → 1.3848 (0.1036; 1.4012) | [run](https://wandb.ai/leena12/drpt_opus/runs/a0y6sott) |
| GlobalOptA | 2.2198 → 1.3388 (0.8811; 1.3766) | 2.0594 → 1.6302 (0.4292; 1.6513) | 1.4926 → 1.3923 (0.1003; 1.4093) | [run](https://wandb.ai/leena12/drpt_opus/runs/kxbmjisz) |
| LayerwiseOptA | 2.2198 → 1.0633 (1.1566; 1.1412) | 2.0594 → 1.5777 (0.4817; 1.5973) | 1.4880 → 1.3840 (0.1041; 1.4007) | [run](https://wandb.ai/leena12/drpt_opus/runs/fvp3qpg5) |
| GlobalSoft | 2.2198 → 1.3552 (0.8646; 1.3888) | 2.0594 → 1.6410 (0.4184; 1.6611) | 1.4855 → 1.3880 (0.0975; 1.4048) | [run](https://wandb.ai/leena12/drpt_opus/runs/swf5af8j) |
| LayerwiseSoft | 2.2198 → 1.0708 (1.1490; 1.1438) | 2.0594 → 1.5783 (0.4811; 1.5969) | 1.4831 → 1.3808 (0.1023; 1.3973) | [run](https://wandb.ai/leena12/drpt_opus/runs/g3zoxl07) |

### alpaca_samsum · Muon (hybrid)

| Method | Target initial → final (reduction; AUC) | General initial → final (reduction; AUC) | Train early → late (reduction; AUC) | W&B |
|---|---:|---:|---:|---|
| FullTraining | 2.2198 → 2.0114 (0.2084; 2.0354) | 2.0594 → 1.9485 (0.1109; 1.9612) | 1.7045 → 1.6360 (0.0685; 1.6475) | [run](https://wandb.ai/leena12/drpt_opus/runs/akvp8g5n) |
| GlobalRaw | 2.2198 → 2.0188 (0.2011; 2.0418) | 2.0594 → 1.9498 (0.1096; 1.9630) | 1.7232 → 1.6601 (0.0631; 1.6708) | [run](https://wandb.ai/leena12/drpt_opus/runs/k2q6jwf4) |
| LayerwiseRaw | 2.2198 → 1.9907 (0.2291; 2.0210) | 2.0594 → 1.9372 (0.1222; 1.9519) | 1.7242 → 1.6607 (0.0636; 1.6716) | [run](https://wandb.ai/leena12/drpt_opus/runs/taxwlwuo) |
| GlobalOptA | 2.2198 → 1.9838 (0.2361; 2.0114) | 2.0594 → 1.9341 (0.1252; 1.9491) | 1.7231 → 1.6606 (0.0624; 1.6710) | [run](https://wandb.ai/leena12/drpt_opus/runs/eta41qsz) |
| LayerwiseOptA | 2.2198 → 1.9827 (0.2372; 2.0139) | 2.0594 → 1.9348 (0.1246; 1.9503) | 1.7227 → 1.6623 (0.0605; 1.6726) | [run](https://wandb.ai/leena12/drpt_opus/runs/zpj6zspu) |
| LayerwiseSoft | 2.2198 → 1.9924 (0.2274; 2.0190) | 2.0594 → 1.9361 (0.1233; 1.9514) | 1.7234 → 1.6600 (0.0633; 1.6709) | [run](https://wandb.ai/leena12/drpt_opus/runs/bcvkxm3c) |
| GlobalMuonSur | 2.2198 → 2.0297 (0.1901; 2.0544) | 2.0594 → 1.9570 (0.1023; 1.9690) | 1.7242 → 1.6630 (0.0612; 1.6736) | [run](https://wandb.ai/leena12/drpt_opus/runs/5d9t2rht) |
| LayerwiseMuonSur | 2.2198 → 1.9876 (0.2322; 2.0159) | 2.0594 → 1.9342 (0.1252; 1.9499) | 1.7233 → 1.6607 (0.0626; 1.6715) | [run](https://wandb.ai/leena12/drpt_opus/runs/d72x20wa) |

### less_squad · AdamW

| Method | Target initial → final (reduction; AUC) | General initial → final (reduction; AUC) | Train early → late (reduction; AUC) | W&B |
|---|---:|---:|---:|---|
| FullTraining | 4.1132 → 3.8879 (0.2253; 3.8676) | 4.8678 → 4.5604 (0.3074; 4.5452) | 1.7459 → 1.4930 (0.2529; 1.5405) | [run](https://wandb.ai/leena12/drpt_opus/runs/pphirn6t) |
| GlobalRaw | 4.1132 → 3.2952 (0.8180; 3.3104) | 4.8678 → 3.9673 (0.9005; 3.9946) | 1.8053 → 1.5649 (0.2404; 1.6118) | [run](https://wandb.ai/leena12/drpt_opus/runs/u3h9vbg8) |
| LayerwiseRaw | 4.1132 → 2.7465 (1.3667; 2.9025) | 4.8678 → 3.4487 (1.4192; 3.6020) | 1.8015 → 1.5593 (0.2422; 1.6056) | [run](https://wandb.ai/leena12/drpt_opus/runs/2v61qk31) |
| GlobalOptA | 4.1132 → 3.1826 (0.9306; 3.2359) | 4.8678 → 3.8477 (1.0202; 3.9005) | 1.8095 → 1.5703 (0.2393; 1.6165) | [run](https://wandb.ai/leena12/drpt_opus/runs/hlh88kef) |
| LayerwiseOptA | 4.1132 → 2.7112 (1.4020; 2.8310) | 4.8678 → 3.3888 (1.4791; 3.5137) | 1.8045 → 1.5606 (0.2439; 1.6070) | [run](https://wandb.ai/leena12/drpt_opus/runs/r0xut9z0) |
| GlobalSoft | 4.1132 → 3.2042 (0.9090; 3.2359) | 4.8678 → 3.8465 (1.0213; 3.8873) | 1.7970 → 1.5521 (0.2449; 1.5996) | [run](https://wandb.ai/leena12/drpt_opus/runs/j077qvb5) |
| LayerwiseSoft | 4.1132 → 2.7110 (1.4022; 2.8373) | 4.8678 → 3.3779 (1.4899; 3.4990) | 1.7952 → 1.5466 (0.2486; 1.5937) | [run](https://wandb.ai/leena12/drpt_opus/runs/9ea6pk6f) |

### less_squad · Muon (hybrid)

| Method | Target initial → final (reduction; AUC) | General initial → final (reduction; AUC) | Train early → late (reduction; AUC) | W&B |
|---|---:|---:|---:|---|
| FullTraining | 4.1132 → 3.8231 (0.2901; 3.8763) | 4.8678 → 4.5729 (0.2949; 4.6286) | 1.9666 → 1.8578 (0.1088; 1.8839) | [run](https://wandb.ai/leena12/drpt_opus/runs/1d3xv69a) |
| GlobalRaw | 4.1132 → 3.7990 (0.3143; 3.8607) | 4.8678 → 4.5563 (0.3115; 4.6171) | 1.9973 → 1.8979 (0.0994; 1.9220) | [run](https://wandb.ai/leena12/drpt_opus/runs/l0gfhxt0) |
| LayerwiseRaw | 4.1132 → 3.7890 (0.3242; 3.8490) | 4.8678 → 4.5354 (0.3324; 4.6018) | 1.9982 → 1.8990 (0.0992; 1.9229) | [run](https://wandb.ai/leena12/drpt_opus/runs/1g1lhlzb) |
| GlobalOptA | 4.1132 → 3.8007 (0.3125; 3.8615) | 4.8678 → 4.5537 (0.3142; 4.6166) | 1.9984 → 1.9025 (0.0960; 1.9259) | [run](https://wandb.ai/leena12/drpt_opus/runs/qyng2zr2) |
| LayerwiseOptA | 4.1132 → 3.8006 (0.3126; 3.8569) | 4.8678 → 4.5504 (0.3175; 4.6106) | 1.9980 → 1.9004 (0.0977; 1.9244) | [run](https://wandb.ai/leena12/drpt_opus/runs/6hl78ku0) |
| LayerwiseSoft | 4.1132 → 3.7894 (0.3238; 3.8492) | 4.8678 → 4.5387 (0.3291; 4.6026) | 1.9980 → 1.8981 (0.0999; 1.9225) | [run](https://wandb.ai/leena12/drpt_opus/runs/9exmssvj) |
| GlobalMuonSur | 4.1132 → 3.8315 (0.2818; 3.8786) | 4.8678 → 4.5749 (0.2929; 4.6317) | 1.9972 → 1.8974 (0.0998; 1.9215) | [run](https://wandb.ai/leena12/drpt_opus/runs/nsj2vt8o) |
| LayerwiseMuonSur | 4.1132 → 3.7966 (0.3166; 3.8563) | 4.8678 → 4.5512 (0.3167; 4.6103) | 1.9978 → 1.8995 (0.0982; 1.9235) | [run](https://wandb.ai/leena12/drpt_opus/runs/husstcne) |

### less_tydiqa · AdamW

| Method | Target initial → final (reduction; AUC) | General initial → final (reduction; AUC) | Train early → late (reduction; AUC) | W&B |
|---|---:|---:|---:|---|
| FullTraining | 1.4197 → 1.1607 (0.2589; 1.1492) | 1.9630 → 1.6174 (0.3456; 1.5852) | 1.7461 → 1.4929 (0.2532; 1.5405) | [run](https://wandb.ai/leena12/drpt_opus/runs/yktaow39) |
| GlobalRaw | 1.4197 → 0.9807 (0.4389; 0.9699) | 1.9630 → 1.2392 (0.7238; 1.2216) | 1.7526 → 1.5103 (0.2424; 1.5556) | [run](https://wandb.ai/leena12/drpt_opus/runs/64jyopv8) |
| LayerwiseRaw | 1.4197 → 0.6968 (0.7228; 0.7545) | 1.9630 → 0.9047 (1.0583; 0.9658) | 1.7463 → 1.5051 (0.2412; 1.5497) | [run](https://wandb.ai/leena12/drpt_opus/runs/feoi981t) |
| GlobalOptA | 1.4197 → 0.8564 (0.5633; 0.8860) | 1.9630 → 1.1272 (0.8358; 1.1438) | 1.7520 → 1.5102 (0.2418; 1.5556) | [run](https://wandb.ai/leena12/drpt_opus/runs/qoidn0bv) |
| LayerwiseOptA | 1.4197 → 0.6655 (0.7541; 0.7273) | 1.9630 → 0.8206 (1.1424; 0.9069) | 1.7468 → 1.5030 (0.2438; 1.5490) | [run](https://wandb.ai/leena12/drpt_opus/runs/0w7u8yvf) |
| GlobalSoft | 1.4197 → 0.9212 (0.4985; 0.9350) | 1.9630 → 1.2347 (0.7283; 1.2178) | 1.7394 → 1.4966 (0.2428; 1.5417) | [run](https://wandb.ai/leena12/drpt_opus/runs/1l44w8aw) |
| LayerwiseSoft | 1.4197 → 0.6279 (0.7917; 0.6946) | 1.9630 → 0.7724 (1.1906; 0.8580) | 1.7393 → 1.4911 (0.2482; 1.5368) | [run](https://wandb.ai/leena12/drpt_opus/runs/74nwvjq8) |

### less_tydiqa · Muon (hybrid)

| Method | Target initial → final (reduction; AUC) | General initial → final (reduction; AUC) | Train early → late (reduction; AUC) | W&B |
|---|---:|---:|---:|---|
| FullTraining | 1.4197 → 1.3259 (0.0938; 1.3432) | 1.9630 → 1.8180 (0.1450; 1.8438) | 1.9665 → 1.8581 (0.1084; 1.8839) | [run](https://wandb.ai/leena12/drpt_opus/runs/3xdyc8pj) |
| GlobalRaw | 1.4197 → 1.3176 (0.1021; 1.3373) | 1.9630 → 1.8062 (0.1568; 1.8377) | 1.9401 → 1.8395 (0.1006; 1.8628) | [run](https://wandb.ai/leena12/drpt_opus/runs/zayswehm) |
| LayerwiseRaw | 1.4197 → 1.3069 (0.1128; 1.3309) | 1.9630 → 1.7931 (0.1699; 1.8288) | 1.9402 → 1.8396 (0.1006; 1.8632) | [run](https://wandb.ai/leena12/drpt_opus/runs/9scivtyp) |
| GlobalOptA | 1.4197 → 1.3146 (0.1051; 1.3347) | 1.9630 → 1.8013 (0.1617; 1.8326) | 1.9405 → 1.8428 (0.0978; 1.8656) | [run](https://wandb.ai/leena12/drpt_opus/runs/xxr4v0vr) |
| LayerwiseOptA | 1.4197 → 1.3097 (0.1099; 1.3294) | 1.9630 → 1.7943 (0.1687; 1.8269) | 1.9402 → 1.8398 (0.1004; 1.8631) | [run](https://wandb.ai/leena12/drpt_opus/runs/7kxymkdf) |
| LayerwiseSoft | 1.4197 → 1.3094 (0.1103; 1.3326) | 1.9630 → 1.7991 (0.1639; 1.8298) | 1.9400 → 1.8386 (0.1015; 1.8625) | [run](https://wandb.ai/leena12/drpt_opus/runs/wuo8l69z) |
| GlobalMuonSur | 1.4197 → 1.3182 (0.1015; 1.3414) | 1.9630 → 1.8128 (0.1502; 1.8441) | 1.9400 → 1.8385 (0.1015; 1.8621) | [run](https://wandb.ai/leena12/drpt_opus/runs/p3kmr91u) |
| LayerwiseMuonSur | 1.4197 → 1.3080 (0.1117; 1.3314) | 1.9630 → 1.7961 (0.1669; 1.8291) | 1.9399 → 1.8387 (0.1012; 1.8620) | [run](https://wandb.ai/leena12/drpt_opus/runs/x0mhts9n) |

### triviaqa_nq · AdamW

| Method | Target initial → final (reduction; AUC) | General initial → final (reduction; AUC) | Train early → late (reduction; AUC) | W&B |
|---|---:|---:|---:|---|
| FullTraining | 4.6787 → 2.5376 (2.1411; 2.5691) | 5.1573 → 2.6064 (2.5510; 2.6264) | 2.0325 → 1.2935 (0.7390; 1.4470) | [run](https://wandb.ai/leena12/drpt_opus/runs/f6mcmua9) |
| GlobalRaw | 4.6787 → 2.3565 (2.3223; 2.4406) | 5.1573 → 2.5054 (2.6519; 2.5547) | 2.2148 → 1.4967 (0.7181; 1.6396) | [run](https://wandb.ai/leena12/drpt_opus/runs/7m9picjh) |
| LayerwiseRaw | 4.6787 → 2.0920 (2.5867; 2.2122) | 5.1573 → 2.4653 (2.6920; 2.5152) | 2.1973 → 1.4527 (0.7446; 1.5967) | [run](https://wandb.ai/leena12/drpt_opus/runs/25qru4y9) |
| GlobalOptA | 4.6787 → 2.3501 (2.3286; 2.4337) | 5.1573 → 2.5128 (2.6445; 2.5630) | 2.2203 → 1.5075 (0.7128; 1.6401) | [run](https://wandb.ai/leena12/drpt_opus/runs/7vyf5tue) |
| LayerwiseOptA | 4.6787 → 2.1181 (2.5606; 2.2242) | 5.1573 → 2.4807 (2.6766; 2.5254) | 2.1994 → 1.4585 (0.7410; 1.5986) | [run](https://wandb.ai/leena12/drpt_opus/runs/q3hl2mjg) |
| GlobalSoft | 4.6787 → 2.3473 (2.3314; 2.4196) | 5.1573 → 2.5061 (2.6512; 2.5496) | 2.2038 → 1.4890 (0.7149; 1.6245) | [run](https://wandb.ai/leena12/drpt_opus/runs/pbztpow6) |
| LayerwiseSoft | 4.6787 → 2.1406 (2.5381; 2.2400) | 5.1573 → 2.4892 (2.6682; 2.5319) | 2.1811 → 1.4532 (0.7279; 1.5909) | [run](https://wandb.ai/leena12/drpt_opus/runs/eg2pezpl) |

### triviaqa_nq · Muon (hybrid)

| Method | Target initial → final (reduction; AUC) | General initial → final (reduction; AUC) | Train early → late (reduction; AUC) | W&B |
|---|---:|---:|---:|---|
| FullTraining | 4.6787 → 3.7251 (0.9536; 3.9350) | 5.1573 → 4.0730 (1.0843; 4.3070) | 4.6288 → 3.3151 (1.3137; 3.6383) | [run](https://wandb.ai/leena12/drpt_opus/runs/1rg0q1dx) |
| GlobalRaw | 4.6787 → 3.8209 (0.8578; 4.0034) | 5.1573 → 4.1836 (0.9738; 4.3894) | 4.6413 → 3.5164 (1.1249; 3.7856) | [run](https://wandb.ai/leena12/drpt_opus/runs/wxpinlmo) |
| LayerwiseRaw | 4.6787 → 3.8008 (0.8779; 3.9938) | 5.1573 → 4.1641 (0.9933; 4.3834) | 4.6422 → 3.5101 (1.1321; 3.7829) | [run](https://wandb.ai/leena12/drpt_opus/runs/bk010yn8) |
| GlobalOptA | 4.6787 → 3.8143 (0.8645; 3.9962) | 5.1573 → 4.1661 (0.9913; 4.3795) | 4.6439 → 3.5300 (1.1139; 3.7928) | [run](https://wandb.ai/leena12/drpt_opus/runs/udqtu1bl) |
| LayerwiseOptA | 4.6787 → 3.8035 (0.8752; 3.9888) | 5.1573 → 4.1694 (0.9879; 4.3814) | 4.6425 → 3.5169 (1.1256; 3.7867) | [run](https://wandb.ai/leena12/drpt_opus/runs/emzc9exu) |
| LayerwiseSoft | 4.6787 → 3.7834 (0.8953; 3.9720) | 5.1573 → 4.1325 (1.0248; 4.3545) | 4.6365 → 3.4665 (1.1700; 3.7464) | [run](https://wandb.ai/leena12/drpt_opus/runs/r1kspihl) |
| GlobalMuonSur | 4.6787 → 3.8064 (0.8723; 3.9955) | 5.1573 → 4.1720 (0.9854; 4.3852) | 4.6426 → 3.5067 (1.1359; 3.7795) | [run](https://wandb.ai/leena12/drpt_opus/runs/c64dbosa) |
| LayerwiseMuonSur | 4.6787 → 3.8054 (0.8733; 3.9989) | 5.1573 → 4.1830 (0.9743; 4.3955) | 4.6455 → 3.5379 (1.1077; 3.7998) | [run](https://wandb.ai/leena12/drpt_opus/runs/75i7gukl) |

## Missing or incomplete requested runs

| Setting | Optimizer | Method | Status | Reason |
|---|---|---|---|---|
| alpaca_samsum | Muon (hybrid) | GlobalSoft | incomplete | missing training completion marker or model.safetensors |
| less_squad | Muon (hybrid) | GlobalSoft | incomplete | missing training completion marker or model.safetensors |
| less_tydiqa | Muon (hybrid) | GlobalSoft | incomplete | missing training completion marker or model.safetensors |
| triviaqa_nq | Muon (hybrid) | GlobalSoft | incomplete | missing training completion marker or model.safetensors |

## W&B panel recipe

In project `leena12/drpt_opus`, filter by group `<setting>-adamw-s42` or `<setting>-hybrid-s42`, then filter run names to the methods in this report. Use `train/global_step` as the x-axis and add panels for `train/val_loss` (target), `eval/loss` (general held-out), and `train/loss`. Keep evaluation curves unsmoothed; smoothing is useful only for `train/loss`. Exact URLs and groups are in `wandb_runs.csv`.
