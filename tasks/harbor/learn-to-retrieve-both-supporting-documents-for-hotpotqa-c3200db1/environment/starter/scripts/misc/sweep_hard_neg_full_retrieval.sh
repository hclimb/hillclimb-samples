# Sweep benchmarks/bench_hard_neg_full_retrieval.py across retrieval modes and batch sizes at the
# multihop_hard_neg_full geometry (num_chunks_per_doc=256, doc_chunk_seq_len=256 -> m_per_query=65,536).
# See wiki/implementations/2026-08-02-hard-neg-full-efficient-retrieval.md.
#
# MUST run on BOTH hosts of the flex slice together (multi-tpu-box-run.sh, same script on each) —
# a single-host launch on a multi-host slice HANGS in libtpu backend init waiting for its peer
# (wiki/infrastructure/experiment-launch-instructions.md §2.3 rule 1). This is a libtpu-level
# peer-wait, independent of jax.distributed.initialize()/JAX_FORCE_SINGLE_HOST (that only gates the
# python-level distributed-system call, not the backend's own topology wait) — cost ~25min of
# wall-clock the first time around (see the implementation note's "hang" entry).
#
# Per-combo timeout so one bad (e.g. full_masked at large B, which is expected to OOM/degrade badly
# at this scale — the whole point of this exploration) config can't block the rest of the sweep.
cd ~/memory-layers
source scripts/infrastructure/setup_shell.sh 2>/dev/null || true

COMBO_TIMEOUT="${SWEEP_COMBO_TIMEOUT:-180}"
MODES="${SWEEP_MODES:-batched two_pass two_pass_kchunk chunked full_masked}"
BATCHES="${SWEEP_BATCHES:-4 8 16 32}"

for MODE in $MODES; do
  for B in $BATCHES; do
    echo "=== MODE=$MODE B=$B ==="
    timeout "$COMBO_TIMEOUT" env MEM_BENCH_MODE=$MODE MEM_BENCH_B=$B .venv/bin/python benchmarks/bench_hard_neg_full_retrieval.py 2>&1 | tail -5
    rc=${PIPESTATUS[0]}
    [ "$rc" = "124" ] && echo "MODE=$MODE B=$B -> TIMED OUT after ${COMBO_TIMEOUT}s"
  done
done
