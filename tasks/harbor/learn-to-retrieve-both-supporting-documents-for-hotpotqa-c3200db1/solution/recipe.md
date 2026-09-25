For a valid submission, progress = (C - B) / (0.715285038072705 - B), where C is candidate mean nDCG@10 and B is the frozen starter remeasured on the same panel. Reward = clip((progress - 0.01) / 0.98, 0, 1). The 1% margin is a fraction of the starter-to-best gap at each end: progress up to 0.01 scores 0 and progress from 0.99 scores 1. Invalid submissions score 0. Every non-smoke quality run measures the starter; --paired remains accepted.

Reference provenance: /data/pranav-work/experiments/reward-ratio-rollouts-20260922-r2/jobs/reward-ratios-fable-3h-20260922T023932Z/learn-to-retrieve-both-supportin__gLoDw6S/waypoints/verification/139f07c6508c8664c007a12e2f81a2bbd29eb32dccdef9b434bb88cc215622e6/record.json

The reference archive installs unchanged source files; it contains no pretrained output weights or compiled binaries. The archive member SHA-256 values are recorded in source-hashes.json.
