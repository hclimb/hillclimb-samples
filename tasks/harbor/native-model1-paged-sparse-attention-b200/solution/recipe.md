Progress = (candidate_rate / baseline_rate - 1) / (1.3515515495176977 - 1). Reward = clip((progress - 0.01) / 0.98, 0, 1). The 1% margin is a fraction of the starter-to-best gap at each end: progress up to 0.01 scores 0 and progress from 0.99 scores 1. The frozen starter is remeasured on the same GPU with matching workloads and seeds. Invalid submissions score 0; baseline or infrastructure failures remain evaluator errors.

Reference provenance: /data/pranav-work/experiments/astra-attention-6h-no-resume-20260923/extra-four/jobs/attention-astra-6h-no-resume-extra4-20260923T025411Z/native-model1-paged-sparse-atten__KeRjELa/result.json

The reference archive installs unchanged source files; it contains no pretrained output weights or compiled binaries. The archive member SHA-256 values are recorded in source-hashes.json.
