# Experiment Launch Instructions

> **Audience: a future agent tasked with kicking off a training or eval run.**
> This is the end-to-end runbook for standing up compute on the TRC TPUs and
> launching an experiment cleanly. Read it fully before provisioning anything.

---

## 0. TL;DR checklist

To cleanly launch a training run:

1. **`.env` complete locally** — it is copied to the box. If a `.env` already exists it
   is probably complete.
2. **Get a box** (§2): use a `READY` persistent `rohun-v6e-8-*`, or provision / keep one
   alive as a spot TPU with tpunanny, or request a flex-start box (§2.2).
   **Want more than one host — e.g. a v6e-8 as 2 × `ct6e-standard-4t`? That is a different
   provisioning path entirely; go straight to §2.3.** A plain `instances create` cannot build a
   slice, and single-host habits (one `TPU_NAME`, one launch) silently break on one.
3. **Launch with `multi-vm-tpu-run.sh`** (§3): set its vars (all env-overridable, no edit
   needed) — especially `RUN_SCRIPT_PATH` (the run `.sh` it executes) and `TPU_NAME`. It tars
   the tree you're standing in (**works from a worktree**; picks up uncommitted edits — **no
   push needed**), copies creds, sets up the venv (**idempotent** — skips rebuild on a
   pre-existing box), and **runs the chosen script** on all workers **inside a detached tmux
   session** — so the run survives the launcher dying. You no longer wrap it in tmux yourself.
   For **several boxes at once** (training + an eval box), use **`multi-tpu-box-run.sh`** (§3.1).
4. **For a manual launch** (SSHing in yourself), run the `.sh` under `tmux`/`nohup` so it
   survives SSH disconnect (§4).
5. **On preemption/reboot**, resume from the latest checkpoint with
   `scripts/misc/resume_launch.sh` (§5). ⚠️ Untested in the current offline-parquet config —
   verify before trusting it.

---

## 1. Compute infrastructure

All runs are on **TRC "trc2"** TPUs.

- **Project:** `memorylayers` (no hyphen — do not confuse with the *separate* GCP projects
  `memory-layers`, `memory-layers-484918`, `memory-layers-data-synthesis`; the flex-start/DWS
  path in §2.2/§2.3 uses `memory-layers`, a different project from this one).
  **Zone:** `europe-west4-a`
- **Accelerator:** `v6e-8` (Trillium) — 8 chips, single-host, ~1.4 TB host RAM,
  31 GB HBM/chip. (Larger pods like `v5litepod-32` appear in some scripts.)
- **Auth:** authenticate `gcloud` locally as **`rohunagrawal@gmail.com`** — this is
  the identity with TRC/TPU-API access. The VM's attached compute SA does **not**
  have TPU scopes. Run `gcloud auth login` if calls fail.
  - tpunanny constructs its API client at import time, so `GOOGLE_APPLICATION_CREDENTIALS`
    must point at the gmail ADC (`~/.config/gcloud/legacy_credentials/rohunagrawal@gmail.com/adc.json`)
    **before** `import tpunanny`.
- **GCS identity** (2026-07-16: now the *same* account — the two-identity split is gone).
  The box reads/writes GCS with the **`GCS_USER_EMAIL` ADC from `.env`**, currently
  **`rohunagrawal@gmail.com`** (verified: it can list `gs://memory-layers-training`).
  Previously this was `ra3440@columbia.edu` while gmail did TPU provisioning.
  - **`.env`'s `GCS_USER_EMAIL` is the single source of truth.** `utils.py::setup_gcs_credentials`
    derives `GOOGLE_APPLICATION_CREDENTIALS` from it, `multi-vm-tpu-run.sh` reads it out of `.env`
    to decide which ADC to copy, and every box/analysis script inherits it. **Never hardcode an
    account** — a script pinning a second opinion silently disagrees with what the box actually
    uses, and you only find out when something dies on `DefaultCredentialsError` mid-run.
  - The launcher now **fails fast** if the ADC for that identity is absent both locally and on the
    box; fix with `gcloud auth login <GCS_USER_EMAIL>` (which writes
    `~/.config/gcloud/legacy_credentials/<email>/adc.json` — note `application-default login`
    writes a *different* path the launcher doesn't read), then re-run.
- **Disk:** root `/` is small (~97 GB); a 4B optimizer-state checkpoint is ~24 GB.
  Stage checkpoints on **GCS** or RAM-backed **`/dev/shm`**, never `/tmp`.

**TPU naming (important):**
- `rohun-v6e-8-*` — persistent / manually-managed boxes (referenced across plans).
- `tn-<tpu_type>-<idx>` — the names tpunanny's `babysit(idxs=...)` auto-generates.
- The authoritative name for any given experiment lives in its launcher script or
  plan file — check there, don't assume.

### Two box shapes — check which one you have before launching

Everything above describes a **Cloud TPU VM** (a TPU-API node). There is a second shape: a
**Compute Engine VM with an attached TPU** (machine type `ct6e-*`), e.g. `tpu-v6e-vm` in
project `memory-layers`. The two are addressed by different `gcloud` surfaces, and the
distinction is invisible until a launch fails:

| | Cloud TPU VM | GCE VM + attached TPU |
|---|---|---|
| Lists under | `gcloud compute tpus tpu-vm list --zone=…` | `gcloud compute instances list` |
| `tpus tpu-vm describe` | works | **`NOT_FOUND`** |
| ssh/scp | `gcloud alpha compute tpus tpu-vm ssh/scp` | `gcloud compute ssh/scp` |
| Launcher setting | `TRANSPORT=tpu` (default) | **`TRANSPORT=gce`** |
| `--worker` | meaningful (multi-host) | n/a — single host |

If `gcloud compute instances list` shows the box but `tpus tpu-vm list` doesn't, it's the
second kind: pass `TRANSPORT=gce` (§3) or every launcher step dies on `NOT_FOUND`.

**Two things to check on a fresh box of this shape** (both bit us standing up `tpu-v6e-vm`):

⚠️ **Disk.** It was provisioned with a **10 GB** root disk — a torch + `vllm-tpu` venv does not
fit. **Grow the disk; don't work around it.** (The `/dev/shm` symlink trick works but is
RAM-backed, so it evaporates on reboot, and `uv venv` refuses to create through a dangling
symlink: `failed to create directory .venv: File exists`.) Resize in the console or with
`gcloud compute disks resize`, then grow the guest filesystem — online, no reboot:

```bash
gcloud compute ssh <box> --zone=<zone> --tunnel-through-iap --command='
  sudo growpart /dev/nvme0n1 1 && sudo resize2fs /dev/nvme0n1p1 && df -h /'
```

The GCP-side resize alone changes nothing inside the guest — `lsblk` shows the 100 GB disk
while `/` stays at 9.9 GB until `growpart`+`resize2fs` run.

⚠️ **Python version.** The box's system Python is **3.10**, below this repo's
`requires-python = ">=3.11"`, so `uv venv` downloads a standalone build — and picks the
**newest** available (3.14), which has no wheels for `torch`/`jax`/`vllm-tpu`. There is no
`.python-version` in the repo to pin it. Create the venv explicitly:

```bash
cd ~/memory-layers && uv venv --python 3.12
```

`multi-vm-tpu-setup.sh` only runs `uv venv` when `.venv/bin/python` is missing, so a venv
created this way is preserved by every later launch.

---

## 2. Get a box

Three ways to get one, on **different compute programmes** — they do not share quota,
naming, or tooling. Pick by which programme has capacity, and by whether you need >1 host:

| | §2.1 tpunanny (TRC spot) | §2.2 flex-start (DWS) | §2.3 flex-start **multi-host slice** |
|---|---|---|---|
| Programme | TRC, project `memorylayers` | DWS, project `memory-layers` | DWS, project `memory-layers` |
| Hosts | 1 (or a real TPU pod) | **1** | **2+**, ICI-wired as one mesh |
| Resource kind | Cloud TPU **queued resource** | **Compute Engine VM** with attached TPU | **bulk-mode MIG** of GCE VMs |
| Created with | tpunanny `babysit` | `gcloud compute instances create` | template + workload policy + MIG resize |
| Lists under | `gcloud compute tpus tpu-vm list` | `gcloud compute instances list` | `... instance-groups managed list-instances` |
| Launcher transport | `TRANSPORT=tpu` (default) | **`TRANSPORT=gce`** (§1) | **`TRANSPORT=gce`**, launched on **every** host |
| Lifetime | indefinite; recreated on preemption | fixed `--max-run-duration`, then self-deletes | same, per host |
| Preemption | yes (spot) | no — but the hard duration cap is absolute | no |

**"v6e-8" is ambiguous — check which one you mean.** The TRC `rohun-v6e-8-*` boxes are **8 chips
on ONE host**. A flex-start v6e-8 is **2 hosts × 4 chips** (§2.3): same chip count, but multi-host,
so it needs the slice provisioning path *and* multi-host launching. `ct6e-standard-8t` (a single
8-chip v6e VM) is **not** a shape the TPU builders programme offers and has never once been
granted — see the table in §2.2.

## 2.1 tpunanny (TRC spot)

You need a `READY` box before launching. Either reuse a persistent `rohun-v6e-8-*`, or
provision/keep a **spot** TPU alive with tpunanny.

`tpunanny` is an external package at **`../tpunanny`** (`github.com/martin-marek/tpunanny`). It
provisions TPUs as **queued resources** and **recreates any that get preempted**.

Install once: `uv pip install -r ../tpunanny/requirements.txt` (needs authenticated `gcloud`).

Core API (`../tpunanny/tpunanny.py`):
- `babysit(idxs, tpu_type, zone, project_id, ssh_script=None, startup_script=None)`
  — keeps multiple TPUs named `tn-<tpu_type>-<idx>` alive, one thread each, forever.
- `_babysit(tpu_id=..., ...)` — a single TPU with an **explicit** name (use this to
  keep a `rohun-*` box alive; see `../tpunanny/babysit_rohun.py`).
- `_recreate(...)` / `_delete_all_suspended(project_id)` — a preempted spot TPU leaves
  a `SUSPENDED`/`FAILED` queued resource that must be deleted before it can be
  recreated. `_delete_all_suspended` clears them project-wide.
- `monitor.py <project_id>` — dashboard of all TPUs/queued-resources in the project.

Recipe in-repo:
- **Keep a fixed named box alive:** `../tpunanny/babysit_rohun.py`.

Gotcha: **set ADC before importing tpunanny** (see §1).

## 2.2 flex-start (Dynamic Workload Scheduler)

A **separate compute programme from TRC/tpunanny** — different project (`memory-layers`),
different quota, and it produces a plain **Compute Engine VM with an attached TPU**, not a Cloud
TPU node. You request capacity; the request sits `PENDING` until it is granted, then the VM
boots (`STAGING` → `RUNNING`) and runs for at most `--max-run-duration`.

**Verified working command** (v6e-8, `europe-west4-a`):

```bash
gcloud compute instances create tpu-v6e-8-flex \
    --zone=europe-west4-a \
    --machine-type=ct6e-standard-8t \
    --provisioning-model=FLEX_START \
    --request-valid-for-duration=2h \
    --max-run-duration=7d \
    --instance-termination-action=DELETE \
    --image-project=ubuntu-os-accelerator-images \
    --image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e \
    --maintenance-policy=TERMINATE \
    --boot-disk-size=200GB \
    --no-address \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --async \
    --metadata=startup-script="echo 'TPU VM Booted'"
```

Machine types: `ct6e-standard-{1t,4t,8t}` = v6e-{1,4,8}. `8t` is 360 vCPU / 1440 GB host RAM.
`--max-run-duration` accepts at least `7d`; only `--request-valid-for-duration` has the 2h cap.

**Use `--async`.** Without it the command BLOCKS until provisioning finishes, which on a queued
flex request can be many minutes — and if you then kill the client on a timeout, the server-side
operation carries on and creates the instance anyway, so you are left unsure whether a box
exists. With `--async` it returns an operation id immediately; poll with the `describe` below.

Verified on a `ct6e-standard-4t` in `europe-west4-a` (reached `RUNNING` ~120 s after the
request): 4 chips under `/dev/vfio`, 194 GB root disk (so `--boot-disk-size` is honoured and no
`growpart` is needed), 708 GB RAM, system Python 3.10, and — with **no external IP** — working
egress to PyPI and huggingface.co through Cloud NAT, with SSH over `--tunnel-through-iap`.

**Four things that are not obvious, each of which fails the create:**

- **`--no-address` is mandatory.** The org policy `constraints/compute.vmExternalIpAccess`
  blocks external IPs project-wide; without the flag the create dies with *"Constraint
  constraints/compute.vmExternalIpAccess violated"*. Nothing needs provisioning to compensate:
  Cloud NAT (`cloud-nat-default-europe-west4`, `ALL_SUBNETWORKS_ALL_IP_RANGES` on the `default`
  network) already provides egress for PyPI/HF/GCS, and SSH goes in over IAP.
- **`--request-valid-for-duration` is capped at 7200s (2h).** Anything larger is rejected. This
  is how long the *request* stays queued waiting for capacity — unrelated to run length.
- **`--boot-disk-size`.** The image defaults to a 10 GB root disk, which cannot hold a
  torch + `vllm-tpu` venv. Set it at create time; growing it later needs a live
  `growpart` + `resize2fs` (§1).
- **`--maintenance-policy=TERMINATE`** is required for TPU VMs.

**`--instance-termination-action=DELETE` means the VM self-destructs** when
`--max-run-duration` expires — the instance and its **boot disk** are gone, taking the venv, the
staged `$GROUND_HF_PARQUET` cache, and any local outputs with them. This is the intended default
(it returns the capacity and stops the disk bill), so **treat every flex box as ephemeral**:
checkpoints and results belong in `gs://`, datasets on the Hub. `STOP` would preserve the disk
but keeps billing it and does not release the reservation.

**Changing `--max-run-duration` after creation:** `gcloud compute instances update` does not
expose the flag; use `set-scheduling`:

```bash
gcloud compute instances set-scheduling tpu-v6e-8-flex --zone=europe-west4-a --max-run-duration=7d
```

It fails with *"resource ... is not ready"* while the instance is `STAGING` — wait for `RUNNING`.
Prefer fixing duration in place over delete-and-recreate: **deleting a `PENDING` request forfeits
its place in the capacity queue**, and capacity is the scarce resource.

### When a flex box never appears

A failed flex provision **deletes itself**, so there is no instance left to inspect and a box
that failed is indistinguishable from one never requested. The only forensic trail is the
operation log:

```bash
gcloud compute operations list --filter="targetLink~<instance-name>" \
  --format="table(operationType,status,insertTime,error.errors[0].code,error.errors[0].message)"
```

**Two error codes, apparently one cause.** Observed across 8 requests in one session:

| shape | zone | `--max-run-duration` | outcome |
|---|---|---|---|
| `ct6e-standard-8t` | europe-west4-a | 24h | `INTERNAL_ERROR` after minutes in `STAGING` |
| `ct6e-standard-8t` | europe-west4-a | 7d | `INTERNAL_ERROR` after minutes in `STAGING` |
| `ct6e-standard-4t` | europe-west4-a | 7d | **RUNNING** in ~120 s |
| `ct5p-hightpu-4t` | europe-west4-b | 7d | `ZONE_RESOURCE_POOL_EXHAUSTED` in ~15 s |
| `ct5p-hightpu-4t` | europe-west4-b | 4h | `ZONE_RESOURCE_POOL_EXHAUSTED` |
| `ct5p-hightpu-4t` | europe-west1-b | 4h | `ZONE_RESOURCE_POOL_EXHAUSTED` |
| `ct5p-hightpu-4t` | europe-west1-c | 4h | `INTERNAL_ERROR` |
| `ct5p-hightpu-4t` | europe-west1-d | 4h | `INTERNAL_ERROR` |
| `ct6e-standard-8t` | europe-west4-a | **4h** | `INTERNAL_ERROR` after ~12 min in `STAGING` |
| `ct5p-hightpu-4t` | us-east5-b, us-east5-c | 7d | failed |
| `ct5p-hightpu-4t` | **us-central1-a** | **7d** | **RUNNING** — granted immediately |

The last three are the same shape and duration requested within a minute of each other, differing
only by zone — and they split across both codes. So **treat `INTERNAL_ERROR` on a flex insert as
"probably no capacity for this shape here"**, not as a bug to report. `ZONE_RESOURCE_POOL_EXHAUSTED`
is the explicit form (fails in seconds, never reaches `STAGING`); `INTERNAL_ERROR` fails later,
after `STAGING`, and is the same practical outcome.

**Zone is the only variable that has ever mattered here.** The same `7d` request that failed in
five zone/shape combinations was granted immediately in `us-central1-a`. As of this writing:
`ct5p-hightpu-4t` (v5p) works in **us-central1-a** and is exhausted in europe-west4-b,
europe-west1-b/c/d and us-east5-b/c; `ct6e-standard-4t` works in **europe-west4-a**, where
`ct6e-standard-8t` has never once been granted.

**Authoritative zone/shape list (TPU builders program, 2026-07-20).** The programme's
recommended flex-start combinations — request these, not other shapes/zones:

| TPU family | Zones | GCE machine type | Max slice |
|---|---|---|---|
| v6e | us-east5-a / us-east5-b, europe-west4-a, southamerica-west1-a | `ct6e-standard-4t` | 8x16 |
| v6e | us-central1-a | `ct6e-standard-4t` | 8x8 |
| v5p | us-east5-a, us-central1-a | `ct5p-hightpu-4t` | 128 |

Note the v6e row is **`4t` only** — `ct6e-standard-8t` is not a recommended shape anywhere,
which matches its 0-for-4 grant record in europe-west4-a (`INTERNAL_ERROR` after ~12 min of
`STAGING`, every time). For >4 v6e chips, that means multiple `4t` hosts, not a bigger single
machine. For v5p note **us-east5-a** (never tried before 2026-07-20) vs us-east5-**b/c**
(always `ZONE_RESOURCE_POOL_EXHAUSTED`) — the `-a` zone is the recommended one.
Also observed 2026-07-19/20: `us-central1-a` v5p capacity is no longer instant — a 7d request
sat `PENDING` for its full 2h validity and expired ungranted; re-issue and keep polling.

**What actually helps**, in order:
1. **Probe a smaller shape in the same zone.** Separates "flags wrong" from "shape unavailable"
   in ~2 minutes — a `4t` came up first try in the zone where every `8t` failed.
2. **Fan out across zones in parallel.** A request costs nothing unless granted, so issue all
   candidate zones at once with `--async` and delete the extras if several land.
3. ~~**Try a shorter `--max-run-duration`.**~~ **Tested and it does not help.** The theory is
   sound — DWS reserves a contiguous block for the whole window, so 4h should be an easier
   scheduling problem than 7d — but measured across both shapes it changed nothing: v5p failed
   identically at 7d and 4h in every zone, and `ct6e-standard-8t` failed identically at 7d, 24h
   and 4h. Request the duration you actually need; do not shorten it hoping for better odds.
4. **Try a different generation.** v5p and v6e draw on separate pools; one being exhausted says
   nothing about the other.

Note that a zone is only usable if the `default` network has **Cloud NAT** in that region — the
org policy blocks external IPs, so without NAT the box has no egress at all. As of 2026-07-20
`europe-west4`, `europe-west1`, `us-central1` and `us-east5` have it (`gcloud compute routers
list` to check); adding another region is a router + NAT pair (§ below).

**Watch it come up** (`PENDING` = waiting for capacity, not broken):

```bash
gcloud compute instances describe tpu-v6e-8-flex --zone=europe-west4-a \
  --format='value(status,scheduling.maxRunDuration.seconds)'
```

## 2.3 Multi-host flex slice (v6e-8 as 2 × `ct6e-standard-4t`)

> **Quick start — a v6e-8 slice, verified working 2026-07-20.** Four steps: policy → template →
> MIG → resize. Details and the reasoning for each flag are below; copy this if you just want the
> box. Granted in ~7 min; both hosts are `FLEX_START` with a 7-day `maxRunDuration`.
>
> ```bash
> P=memory-layers; R=europe-west4; Z=europe-west4-a; MIG=tpu-v6e-slice-mig
> # 1. topology (2x4 = 8 v6e chips = 2 hosts x 4)      2. flex-start template
> gcloud compute resource-policies create workload-policy v6e-slice-2x4-ew4 \
>     --project=$P --region=$R --type=HIGH_THROUGHPUT \
>     --accelerator-topology=2x4 --accelerator-topology-mode=AUTO_CONNECT
> gcloud compute instance-templates create tpu-v6e-4-flex-tmpl \
>     --project=$P --instance-template-region=$R \
>     --machine-type=ct6e-standard-4t --provisioning-model=FLEX_START \
>     --max-run-duration=7d --instance-termination-action=DELETE \
>     --image-project=ubuntu-os-accelerator-images \
>     --image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e \
>     --maintenance-policy=TERMINATE --boot-disk-size=200GB --no-address \
>     --scopes=https://www.googleapis.com/auth/cloud-platform
> # 3. bulk-mode MIG bound to the policy                4. ask for both hosts at once
> gcloud compute instance-groups managed create $MIG --project=$P --zone=$Z --size=0 \
>     --template=projects/$P/regions/$R/instanceTemplates/tpu-v6e-4-flex-tmpl \
>     --target-size-policy-mode=bulk --default-action-on-vm-failure=do-nothing \
>     --workload-policy=projects/$P/regions/$R/resourcePolicies/v6e-slice-2x4-ew4
> gcloud compute instance-groups managed resize $MIG --project=$P --zone=$Z --size=2
> # watch: PENDING -> STAGING -> RUNNING
> gcloud compute instance-groups managed list-instances $MIG --project=$P --zone=$Z
> ```
>
> Then, on **each** host once: `cd ~/memory-layers && uv venv --python 3.12` (§1), and always
> launch on **both** workers with `multi-tpu-box-run.sh` — see "Launching on it" below.
> **Tear down** with `... instance-groups managed delete $MIG` (deleting the VMs individually
> leaves the MIG, which recreates them).

Everything above provisions **one** VM. A **multi-host slice** — several hosts wired over ICI so
JAX sees one mesh — needs a different path, and the obvious routes are all dead ends:

| Attempt | Why it fails |
|---|---|
| `instances create --resource-policies=<workload policy>` | 400: *"Resource policy must have one of [VmMaintenancePolicy, GroupPlacementPolicy, InstanceSchedulePolicy]"* — a **workload** policy can't attach to a single instance |
| `instances bulk create --provisioning-model=FLEX_START` | *"Invalid choice: 'FLEX_START'"* — bulk create offers only `RESERVATION_BOUND`/`SPOT`/`STANDARD`, on GA **and** alpha/beta |
| MIG with a TPU topology, default target-size mode | *"MIGs with TPU accelerator topology require BULK mode in target size policy"* |
| `resize-requests create` (the usual DWS bulk path) | *"Creating resize requests for managed instance group with TPU is not allowed"* |

**What works: flex-start lives in the instance _template_, and a bulk-mode MIG instantiates it.**
Verified 2026-07-20 — granted in ~7 min, both hosts `FLEX_START` + 7d `maxRunDuration`:

```bash
# 1. Workload policy = the ICI topology. 2x4 = 8 v6e chips = 2 hosts x 4.
#    AUTO_CONNECT pre-forms the topology (PROVISION_ONLY leaves chips unconnected).
gcloud compute resource-policies create workload-policy v6e-slice-2x4-ew4 \
    --project=memory-layers --region=europe-west4 \
    --type=HIGH_THROUGHPUT --accelerator-topology=2x4 --accelerator-topology-mode=AUTO_CONNECT

# 2. Regional template carrying FLEX_START (max-run-duration is REQUIRED with it).
gcloud compute instance-templates create tpu-v6e-4-flex-tmpl \
    --project=memory-layers --instance-template-region=europe-west4 \
    --machine-type=ct6e-standard-4t --provisioning-model=FLEX_START \
    --max-run-duration=7d --instance-termination-action=DELETE \
    --image-project=ubuntu-os-accelerator-images \
    --image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e \
    --maintenance-policy=TERMINATE --boot-disk-size=200GB --no-address \
    --scopes=https://www.googleapis.com/auth/cloud-platform

# 3. Zonal MIG in BULK mode bound to the policy, then resize to the host count.
#    Bulk mode waits for capacity for ALL VMs and creates them together = one slice.
gcloud compute instance-groups managed create tpu-v6e-slice-mig \
    --project=memory-layers --zone=europe-west4-a \
    --template=projects/memory-layers/regions/europe-west4/instanceTemplates/tpu-v6e-4-flex-tmpl \
    --size=0 --target-size-policy-mode=bulk \
    --workload-policy=projects/memory-layers/regions/europe-west4/resourcePolicies/v6e-slice-2x4-ew4 \
    --default-action-on-vm-failure=do-nothing
gcloud compute instance-groups managed resize tpu-v6e-slice-mig \
    --project=memory-layers --zone=europe-west4-a --size=2
```

The `--workload-policy` value must be a **full URL**; a bare name gives *"The URL is malformed."*

**Confirm it's a real slice, not two co-located VMs** — check both sides:

```bash
gcloud compute instance-groups managed describe tpu-v6e-slice-mig --zone=europe-west4-a \
  --format='yaml(status.appliedAcceleratorTopologies)'     # -> acceleratorTopology: 2x4, state: ACTIVE
# and on the box:
curl -s -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/tpu-env
# -> TPU_ACCELERATOR_TYPE: 'v6e-8'  TOPOLOGY: '2x4'  TPU_WORKER_ID: '0'  HOST_BOUNDS: '1,2,1'
```

**Watch the queue:** `gcloud compute instance-groups managed list-instances tpu-v6e-slice-mig
--zone=… ` — VMs sit `PENDING` → `STAGING` → `RUNNING`. `instances list` shows them too.

**Launching on it — two rules:**

1. **Run on EVERY worker, always.** A single-host launch **hangs** at
   `TPU backend initialization is taking more than 60.0 seconds. Did you run your code on all TPU
   hosts?` — the backend blocks waiting for its peer. Use `multi-tpu-box-run.sh` with the *same*
   script on both (§3.1), with `TRANSPORT=gce`:

   ```bash
   TRANSPORT=gce ZONE=europe-west4-a PROJECT_ID=memory-layers \
   bash scripts/infrastructure/multi-tpu-box-run.sh \
     tpu-v6e-slice-mig-1wjb=scripts/embed/<script>.sh \
     tpu-v6e-slice-mig-1z9d=scripts/embed/<script>.sh
   ```

2. **`init_jax_distributed()` must actually initialize here** — it does since 2026-07-20; older
   trees do not and **will fail**. The slice publishes `worker-network-endpoints` as bare
   comma-separated instance names (`tpu-v6e-slice-mig-1wjb,tpu-v6e-slice-mig-1z9d`, no colons),
   which the old colon-format check read as single-host, skipping init. `utils.py` now splits on
   the endpoint **count** and initializes explicitly for 2+
   ([implementation note](../implementations/2026-07-20-multihost-gce-slice-distributed-init.md)).

   ⚠️ **A healthy `device_count` does NOT mean the distributed system is up.** libtpu forms the
   ICI mesh on its own, so with init skipped a probe still shows `process_count=2`,
   `device_count=8`, `local_device_count=4` and the full 2x4 coords — everything looks fine. The
   failure lands much later, in orbax at `setup_checkpointing`:
   `ValueError: Distributed system is not available; please initialize it via
   jax.distributed.initialize()`. This cost a training launch. When bringing up a new box shape,
   probe the **coordination** path (`JAX_PROBE_DISTRIBUTED=1
   scripts/infrastructure/probe_jax_slice.sh` — it runs a `process_allgather` and a checked global
   collective), not just device discovery.

   Harmless quirk: `jax.process_index()` does **not** match the `process_id` passed in (JAX ranks
   in its own order — here 1wjb→1, 1z9d→0, inverted vs metadata). Verified coherent: local device
   sets are disjoint and complete (`[0-3]`/`[4-7]`), allgather sees `[0,1]`, and a global
   collective is correct. Exactly one process is rank 0, which is all the code requires.

3. **Evals on a slice** (supported since 2026-07-20 — the whole eval path needed multi-host
   fixes; see the [implementation note](../implementations/2026-07-20-rag-hybrid-evaluator.md)):
   - Only the **JAX rank-0** host writes results/manifest and runs the vLLM judge — and rank
     order is NOT host order (see the inversion quirk above). The other host prints
     `no manifest on this host` and exits 0; that is correct behavior, not a failure.
   - The judge needs `VLLM_TPU_LOCAL_ONLY=1` (set by slice runner scripts): it confines the
     vLLM subprocess to the host's own 4 chips. Without it, libtpu reads the slice topology
     from instance metadata and the engine proc dies waiting for its peer
     ("Engine core initialization failed", then a 1800 s server-ready timeout).
   - Upload/wandb steps must be gated on the results file existing locally (the runner
     scripts use a `find … | grep -q .` guard) so they self-select the rank-0 host.

### Ops lessons from the 2026-07-20 slice bring-up (silent process kills)

- **If TPU processes die silently (no traceback, no OOM, tmux sessions vanish): suspect a
  kill loop from ANOTHER machine before blaming the box.** The bring-up lost hours to an
  orphaned cleanup loop on the dev box —
  `until gcloud ssh <box> --command='tmux kill-server; pkill -9 -f venv/bin/python'; do …` —
  whose `pkill -f` pattern matched its own remote wrapper shell, so every strike killed its
  own ssh (rc 255) and the loop retried forever, firing every ~2.5 min. Check
  `ps aux | grep gcloud` on machines with credentials, and on the box identify senders with
  zero installs via ftrace:
  `echo 'sig == 9' > /sys/kernel/tracing/events/signal/signal_generate/filter` +
  `echo 1 > …/enable`, reproduce, read `…/trace` (sender comm/pid in the TASK column; walk
  parents via the `sched_process_fork`/`exec` events).
- **`ls /dev/vfio/` empty ⇒ the TPU devices got unbound** (seen after repeated SIGKILLs of
  live libtpu processes); a guest reboot restores them. Corollary: never `timeout`/SIGKILL a
  process that has touched libtpu — let run scripts' own cleanup handle stale workers.
- A wedged post-OOM chip state also fails `probe_jax_slice.sh` (passed at 12:24, failed at
  20:23 the same day after OOM-heavy runs); reboot both hosts, re-probe, relaunch.

Per-host setup is unchanged from §1: system Python is 3.10, so `uv venv --python 3.12` on **each**
host before the first launch. First result off this shape:
[2026-07-20 approx top-k on the grounding 4-layer config](../experiments/2026-07-20-ground4layer-approx-topk-v6e-slice.md).

### Adding a new region (Cloud NAT)

A flex box in a region without Cloud NAT on the `default` network comes up with **no internet
egress** — no PyPI, no HF, no GCS. Two commands, plus Private Google Access so GCS traffic takes
Google's private path instead of paying NAT bandwidth for every checkpoint read:

```bash
gcloud compute routers create router-<region> --network=default --region=<region>
gcloud compute routers nats create cloud-nat-default-<region> \
    --router=router-<region> --region=<region> \
    --auto-allocate-nat-external-ips --nat-all-subnet-ip-ranges     # note: IPs, plural
gcloud compute networks subnets update default --region=<region> --enable-private-ip-google-access
```

Keep the box in **`europe-west4` or as near as possible**: `gs://memory-layers-training` is a
*regional* bucket in `EUROPE-WEST4`, so a US zone means transatlantic reads of ~24 GB
checkpoints. `europe-west1` is the nearest region that offers v5p.

**Then launch against it with `TRANSPORT=gce`** — it is a GCE VM, so the Cloud TPU API cannot see
it and the default `tpu-vm ssh` path returns `NOT_FOUND` (§1):

```bash
TPU_NAME=tpu-v6e-8-flex ZONE=europe-west4-a PROJECT_ID=memory-layers TRANSPORT=gce \
RUN_SCRIPT_PATH=scripts/embed/train_musique_sft_midtrain.sh \
  bash scripts/infrastructure/multi-vm-tpu-run.sh
```

Remember the Python-version gotcha from §1: system Python is 3.10, below this repo's
`requires-python`, so create the venv explicitly with `uv venv --python 3.12` before the first
launch or `uv` picks 3.14 and no ML wheels exist for it.

---

## 3. Launch with `multi-vm-tpu-run.sh`

This is the launch path: it syncs your code to a known-good box/pod **and runs your
chosen run script** on it. Every setting is a variable with a default **and is overridable
from the environment** (no need to edit the file) — `TPU_NAME`, `ZONE`, `PROJECT_ID`,
**`RUN_SCRIPT_PATH`** (the `.sh` it executes), `GCS_USER_EMAIL`, `GCLOUD_USERNAME`, `WORKER`:

```bash
TPU_NAME=rohun-v6e-8-0 RUN_SCRIPT_PATH=scripts/embed/bench_approx_topk.sh \
  bash scripts/infrastructure/multi-vm-tpu-run.sh
```

**Worktree-aware:** it derives paths from git, so it's correct whether you run it from the
main checkout or a worktree:
- `REPO_ROOT` = `git rev-parse --show-toplevel` — the tree you're standing in → what it tars.
- `.env` is pulled from the **main checkout** (`dirname $(git rev-parse --git-common-dir)`),
  because worktrees don't inherit the gitignored `.env`.
- It **fails fast** if `RUN_SCRIPT_PATH` isn't in the synced tarball (catches syncing the
  wrong tree — e.g. a worktree-only script that a main-checkout sync would have missed).

Steps it performs (`scripts/infrastructure/multi-vm-tpu-run.sh`):
1. Tars **tracked + untracked** files (`git ls-files --cached --others --exclude-standard`)
   from `REPO_ROOT` → `/tmp/memory_layers_sync.tar.gz`. Picks up uncommitted edits — no push.
2. `scp`s the ADC (`adc.json`, **best-effort** — skipped with a warning if the local ADC is
   absent; the box may already have it), `.env`, the setup script, and the tarball to workers.
3. Runs `scripts/infrastructure/multi-vm-tpu-setup.sh` — **idempotent**: installs uv only if
   missing, extracts the code **preserving `.venv`/`.git`**, creates the venv only if absent,
   `uv pip install -e .` only when `pyproject.toml`/`uv.lock` changed (sha256 marker in the
   venv), and `wandb login` only if not already authenticated. On a pre-existing box this is a
   fast code re-sync, not a full rebuild.
4. SSHes all workers to `source scripts/infrastructure/setup_shell.sh && cd memory-layers && source $RUN_SCRIPT_PATH`
   — i.e. it launches `RUN_SCRIPT_PATH` on the box.

Notes:
- **The run is detached by default** (`DETACH=1`): step 4 hands the script to
  `tmux_launch.sh`, which starts it in a tmux session **on the box** and appends a final
  `__RUN_EXIT__=<rc>` line to `~/runs/<session>.log`. The launcher then tails that log, so you
  still get live output and the real exit code — but **Ctrl-C (or the launcher dying) only stops
  the tail, not the run**. Knobs: `DETACH=0` (legacy single blocking SSH), `FOLLOW=0` (start and
  return immediately), `TMUX_SESSION=<name>` (default `run-<script basename>`).
  Re-attach: `... ssh <box> --command='tmux attach -t <session>'`, or `tail -f ~/runs/<session>.log`.
  - **`exec` in a run script is fine.** Several box scripts end in `exec python …`
    (`hard_neg_eval_box_run.sh`, `ground_box_run.sh`, `box_run.sh`), which replaces the shell.
    `tmux_launch.sh` therefore sources the script inside a **subshell**, so `exec` replaces only
    that subshell and the outer shell survives to write `__RUN_EXIT__=$?` (still the exec'd
    program's real code). Without the subshell the sentinel never lands and `FOLLOW` tails
    forever — an orphaned `tail -f` long after the run died.
- Multi-host mechanics: `train.py` calls `jax.distributed.initialize()`; logging/wandb is
  guarded by `jax.process_index()==0`.

## 3.1 Several boxes at once — `multi-tpu-box-run.sh`

One script per box, all launched in parallel (e.g. a training run + its
[eval box](../evaluation/eval-boxes.md), which **must** be a separate box — the judge's vLLM
needs the TPU that training is holding):

```bash
bash scripts/infrastructure/multi-tpu-box-run.sh \
    rohun-v6e-8-0=scripts/embed/train_hard_neg_think.sh \
    rohun-v6e-8-1=scripts/embed/hard_neg_eval_box_run.sh
```

Each pair is `<tpu>=<repo-relative-run-script>`; output is line-prefixed with the box name and the
exit code of every box is reported (overall rc = first failure). It calls `multi-vm-tpu-run.sh`
per box, so each run still lands in its own detached tmux session.

**`RUN_START_TIME` is what makes the parallel launch possible.** A run's identity is its run-dir
`<run_name>-<YYYY-MM-DD>-<HH-MM-SS>` (`utils.py::run_dir_name`), which also fixes its wandb id —
and normally `train.py` mints the timestamp at startup, so an eval box can't know the dir until
training prints it (a serial launch). `multi-tpu-box-run.sh` mints **one** UTC timestamp, prints
it, and forwards it to every box, so training pins it via `trainer.run_start_time` and the eval box
composes the same `RUN_DIR`. Override it to re-attach boxes to an existing run. Reusing a value
with the same `run_name` re-creates the collision run-dirs exist to prevent
([../evaluation/eval-boxes.md](../evaluation/eval-boxes.md)).

**Passing your own variables to a box-side script:** ssh forwards no environment, so use
`RUN_ENV="KEY=VAL KEY=VAL"` on `multi-vm-tpu-run.sh`. It reaches the script via `tmux_launch.sh`'s
trailing `KEY=VAL` args, exported *after* `setup_shell.sh` (so a forwarded value beats `.env`).
Works on the `DETACH=0` path too. `multi-tpu-box-run.sh` uses this to carry `RUN_START_TIME`.

> **Don't just `&` two `multi-vm-tpu-run.sh` calls.** That script tars to a *fixed*
> `/tmp/memory_layers_sync.tar.gz`, so concurrent invocations race on one file and can ship a
> half-written tarball. `multi-tpu-box-run.sh` gives each box a private `TARBALL` directory (the
> basename must stay `memory_layers_sync.tar.gz` — `multi-vm-tpu-setup.sh` matches on it).

---

## 4. Launching a run script manually — best practices

Run scripts can live anywhere (`scripts/`, an ad-hoc script you
write). When you SSH into a box and launch one yourself, rather than via §3:

- **Source the shell env first:** `source scripts/infrastructure/setup_shell.sh` (loads `~/.env`, sets
  `HF_TOKEN`, `HYDRA_FULL_ERROR=1`).
- **Run under `tmux` or `nohup ... &`** so the run survives SSH disconnect.
- A well-formed run script sources `.env`, sets a `CHECKPOINT_DIR`, and calls `train.py`
  or `eval.py` with Hydra overrides — mirror the existing `scripts/embed/*.sh` shape.
- **Smoke-test before walking away:** steps advance (not stuck at compile/step 0), a
  checkpoint lands in GCS, and wandb shows the run (process 0 only).

Example (launch a stage, resuming if a checkpoint exists — see §5):

```bash
cd $HOME/memory-layers && source scripts/infrastructure/setup_shell.sh
bash scripts/misc/resume_launch.sh train_ground_s1.sh ground_s1_zeroinit_4layer
```

`resume_launch.sh` launches in a `tmux` session named `train`.

---

## 5. Checkpoints, preemption & resume

- **Run-dir layout:** `gs://$GCS_BUCKET/<run_name>-<date>-<time>/<model_name>/<step>/`.
- **Full resume** (model + optimizer + dataloader position) uses
  **`scripts/misc/resume_launch.sh <stage_script> <run_name>`**, run **on the box** after a
  clean reboot. It finds the newest run-dir with a checkpoint and sets
  `RESUME_FROM=<that run's model directory>` (the `qwen3_mem_embed` dir, **not** dir/step);
  the trainer restores the latest step + optimizer + dataloader state.
- A resumed run gets a **new run-dir**; eval boxes resolve each step to the newest dir
  that has it, so curves stay continuous.
- ⚠️ **Streaming-path only:** the entire fast-forward / SKIP_LOADER_RESTORE story below applies
  only to `dataset.storage: streaming` (default). On the indexed path
  (`storage: arrayrecord`, [../data/storage-modes.md](../data/storage-modes.md)) the sampler
  seeks to `step * batch_size` in O(1), the trainer skips loader save/restore under
  `is_indexed=True`, and `SKIP_LOADER_RESTORE` is a no-op — remove it from indexed launchers.

- ⚠️ **Dataloader fast-forward is far more expensive than "a few minutes" — budget hours, and
  it can OOM the box.** Measured 2026-07-20 resuming `ground_s1_zeroinit_4layer` @ 38000 on the
  v6e-8 slice (708 GB host RAM, `qa_hard_neg_think_sft4b`, `num_workers: 16`): **~2 h without
  ever producing a first batch**, then killed to pre-empt an OOM. Two compounding costs:

  1. **The saved cursor is a raw-item COUNT, not a seek.** `dataloader_state.json` holds
     `{"count": 716641}` per worker (16 of them, 670k–960k each), and restore *replays* every one
     of those ~13 M items through parquet-read → normalize → tokenize, discarding all of it, to
     get back into position. There is no skip-ahead.
  2. **Each grain worker costs ~40 GB RSS** (own pipeline + the 100 000-item shuffle buffer) and
     they spawn lazily, roughly one per 12 min. Host RAM went 221 → 469 GB with only ~8 of 16 up;
     the rest needed ~320 GB against ~232 GB free. Symptom before the kill: **`sshd` starts
     refusing connections (`exited with return code [255]`)** under the memory pressure.

  **Mitigations**, cheapest first: accept a fresh stream (restore is warn-only — with no state
  file you simply get "stream starts from 0"); lower `dataset.num_workers` (fixes the OOM but
  *lengthens* the replay); or make the cursor seekable (shard + row instead of a count) — the real
  fix, still **open**.

- ⚠️ **Killing a run does NOT kill its data workers.** `tmux kill-server` + `pkill -f train.py`
  leaves the grain workers **orphaned to `ppid=1`**, still holding hundreds of GB — and they never
  match a `train.py` pattern. Symptoms: memory stays pinned, SSH stays flaky.

- ⚠️ **`pkill python3` MISSES the venv interpreter.** `~/memory-layers/.venv/bin/python` has
  `comm` = **`python`**, not `python3`, so `pkill -9 python3` skips it *and* `pgrep -c python3`
  returns 0 — the box looks clean while a process still holds all four chips, and the next launch
  dies with `ABORTED: The TPU is already in use by process with pid …`. Match on the path and
  verify against the device, not a process name:

  ```bash
  pkill -9 -u $(id -u) -f venv/bin/python
  pgrep -u $(id -u) -f venv/bin/python | wc -l     # want 0
  sudo fuser /dev/vfio/0                            # want NO holder
  ```

  (`ps -eo pid,comm,args | grep -i python` will still show `networkd-dispatcher` and
  `unattended-upgrades` — those are system daemons, not yours.)

- ⚠️ **Never import JAX unguarded on ONE host of a multi-host slice** — it blocks in TPU backend
  init waiting for its peer, and killing the ssh leaves the remote process alive still holding the
  TPU. For CPU-only probes (optimizer state shapes, config composition) use `JAX_PLATFORMS=cpu`.

- ⚠️ **Still untested:** whether the offline-parquet stream order matches the live-HF order a run
  originally consumed. Matching the shard *set* (`GROUND_DATA_FRAC=1.0` for a run that streamed
  live HF) is necessary but not sufficient — the count is only meaningful against an identical
  ordering. Restore failure is **warn-only**, so a mismatch silently restarts the data rather than
  erroring. Always check for **"Restored dataloader state"** vs **"stream starts from 0"**.

See [checkpointing.md](checkpointing.md) for the Orbax internals.

---

## Pushing to GitHub (git auth)

`origin` is HTTPS (`https://github.com/rohunagrawal/memory-layers`) and this environment has
**no git credential helper**, so a bare `git push` fails with `could not read Username`.
Authenticate with the **`GIT_PAT`** from `.env`. Note: git **worktrees do not inherit the
gitignored `.env`**, so it lives in the *main* checkout — `/home/rohunagrawal/memory-layers/.env`
— even when you're pushing from a worktree.

Push without persisting the token (keeps it out of `git config` and the command's argv):

```bash
# GIT_PAT lives in the MAIN checkout's .env (worktrees don't inherit it)
export TOKEN="$(grep -E '^(export )?GIT_PAT=' /home/rohunagrawal/memory-layers/.env \
  | head -1 | cut -d= -f2- | tr -d '\"')"
git -c credential.helper='!f(){ echo username=x-access-token; echo "password=$TOKEN"; }; f' \
    push -u origin <branch>
```

- **Never** commit the token or bake it into the `origin` URL / `.git/config` — the inline
  helper above avoids both.
- Do **not** push to `main`/`master`, force-push, or merge from here.

---

## Key files

| File | Role |
|------|------|
| `../tpunanny/tpunanny.py` | babysit / recreate / delete queued spot TPUs |
| `../tpunanny/babysit_rohun.py` | keep one explicitly-named box alive |
| `../tpunanny/monitor.py` | project-wide TPU status dashboard |
| `scripts/infrastructure/multi-vm-tpu-run.sh` | sync local code + launch the chosen run script on a box/pod (detached in tmux) |
| `scripts/infrastructure/multi-tpu-box-run.sh` | launch one script per box across several boxes in parallel |
| `scripts/infrastructure/tmux_launch.sh` | box-side: start a run script in a detached tmux session + exit sentinel |
| `scripts/misc/resume_launch.sh` | resume a run from latest checkpoint on a clean box |

---

*Grounded in the repo + tpunanny as of this writing. If a script's top-of-file vars
(TPU names, zone, branch) disagree with this doc, the script is authoritative — update
this doc.*
