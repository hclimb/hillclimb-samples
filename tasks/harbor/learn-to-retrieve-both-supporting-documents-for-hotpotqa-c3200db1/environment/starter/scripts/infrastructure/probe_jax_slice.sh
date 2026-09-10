#!/usr/bin/env bash
# Diagnostic: what does JAX see on this box, and does distributed init behave?
# Written for the flex-start MULTI-HOST GCE slice (2x ct6e-standard-4t under a 2x4 workload
# policy), a box shape utils.py::init_jax_distributed predates — it treats every GCE-attached
# TPU as single-host. Prints the raw metadata it keys off, then the resulting device topology.
# Read-only; touches no training code. Run on EVERY worker of the slice at once.
#
#   TPU_NAME=tpu-v6e-slice-mig-1wjb ZONE=europe-west4-a PROJECT_ID=memory-layers TRANSPORT=gce \
#   RUN_SCRIPT_PATH=scripts/infrastructure/probe_jax_slice.sh \
#     bash scripts/infrastructure/multi-vm-tpu-run.sh
MD="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
echo "=== host $(hostname) ==="
echo -n "worker-network-endpoints: "; curl -s -H "Metadata-Flavor: Google" "$MD/worker-network-endpoints"; echo
echo -n "agent-worker-number: ";      curl -s -H "Metadata-Flavor: Google" "$MD/agent-worker-number"; echo
echo -n "accelerator-type: ";         curl -s -H "Metadata-Flavor: Google" "$MD/accelerator-type"; echo

# JAX_PROBE_DISTRIBUTED=1 -> exercise the repo's own init path; unset -> raw jax, no init.
uv run python -c '
import os, sys
if os.environ.get("JAX_PROBE_DISTRIBUTED") == "1":
    sys.path.insert(0, os.getcwd())
    from utils import init_jax_distributed
    init_jax_distributed()
import jax
print("[probe] jax", jax.__version__)
print("[probe] process_index/count:", jax.process_index(), "/", jax.process_count())
print("[probe] device_count:", jax.device_count(), " local_device_count:", jax.local_device_count())
print("[probe] devices:", jax.devices())
print("[probe] LOCAL device ids:", sorted(d.id for d in jax.local_devices()))

# Is the mesh actually usable? Device ownership being self-consistent is not the same as
# collectives working. Only run when distributed init was exercised.
if os.environ.get("JAX_PROBE_DISTRIBUTED") == "1":
    import numpy as np, jax.numpy as jnp
    from jax.experimental import multihost_utils
    # 1. Every rank present exactly once, and each rank agrees on who it is.
    seen = multihost_utils.process_allgather(jnp.array([jax.process_index()]))
    print("[probe] allgather(process_index) =", np.asarray(seen).ravel().tolist())
    # 2. A global collective over ALL 8 chips: sum of per-device constants must be 8*(0..7 sum).
    #    A broken mesh gives a wrong total or hangs here rather than at init.
    x = jnp.arange(jax.device_count(), dtype=jnp.float32)
    total = float(jax.jit(lambda a: jnp.sum(a))(x))
    expect = float(sum(range(jax.device_count())))
    verdict = "OK" if total == expect else "MISMATCH"
    print("[probe] global psum: got", total, "expect", expect, "->", verdict)
'
