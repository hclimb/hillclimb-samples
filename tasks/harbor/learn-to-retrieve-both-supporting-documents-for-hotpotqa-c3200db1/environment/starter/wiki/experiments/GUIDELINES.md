# Experiment write-up — guidelines

What each `wiki/experiments/*.md` write-up should cover. These are guidelines, not a rigid
form. Lead with the conclusion: a colleague should be able to reproduce the run and reach the
same answer from the doc alone.

A good write-up makes the following clear:

- **Conclusion, first** — did the hypothesis hold, what do we now believe, and what's the next
  step? Put the headline number here, before the narrative.
- **Hypothesis & motivation** — what you expected and why, and which decision or open question
  the experiment informs.
- **Setup** — model / variant (which layers, bank size, …), data (dataset, split, doc corpus),
  the independent variable(s) swept, what's held fixed / the baseline arm, and the metric(s) and
  how they're judged (LLM judge, NLL, RULER, …).
- **Reproducibility** — enough for a blind rerun: the exact `uv run …` command + Hydra
  overrides, the commit SHA, checkpoint `gs://` path(s), the wandb run URL/ID, and the TPU type
  (e.g. `v4-8`). Note anything non-obvious (env vars like `MEM_TOP_K=…`, judge-server setup,
  one-off patches). The fully-resolved config is also saved in each checkpoint's
  `.hydra/config.yaml`.
- **Results, in full** — don't summarize away the numbers; prefer a Markdown table per arm /
  metric. Keep results git-friendly: metrics/tables in the doc or as committed CSV/JSON; large
  artifacts as pointers, never committed binaries — plots in `results/figures/…`, raw dumps /
  checkpoints as `gs://…`, curves as a wandb URL.
- **Interpretation** — why the numbers came out this way; caveats, confounds, sample-size
  limits. Say so plainly if the result is inconclusive rather than forcing a conclusion.

Head the write-up with its **date, author, status** (running / done / inconclusive).
