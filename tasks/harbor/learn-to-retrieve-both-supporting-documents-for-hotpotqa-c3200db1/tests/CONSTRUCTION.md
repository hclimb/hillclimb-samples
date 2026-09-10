# Candidate status

This is a separate BERT-Tiny candidate, not an update to the released task.
Pinned human-2 takes precedence over human-1 and the historical Qwen plan.
`adaptation_manifest.json` records the changed boundaries and data provenance.

## Layout

`plan.json` declares schema v3, the pinned reconstructed repository and the minimal
training overlay. The public dependency closure is `evaluator/public_tests`. The
compiler copies this declared `public_tests_dir` directly to
`/environment/starter/optifine_public_tests`. Do not duplicate it in the overlay:
public tools and assets belong to the compiler's explicit public packaging stage,
not the repository checkout whose starter digest it records.
The training/reference source files
are separate: `starter-overlay/training.py` is the frozen starter implementation,
while `solution/solve.sh` installs `solution/training.py`. No trained weights ship.

Only `evaluator/private_panel` contains private inputs. Do not copy it, the reference
solution, revision context, evidence or full upstream caches into the solver image.
The build-time `package.py` uses an explicit allowlist of text files and Tiny assets.
It reproduces the public helper closure from the trusted implementation, rather than
maintaining divergent public and private evaluation code. Run packaging after changes.

The existing source repository is 8.9 MB. Its native paragraph rendering and row-ID
semantics are preserved, but the approved fixed-weights boundary deliberately replaces
its TPU retrieval implementation. Public-only qualification uses the new overlay and
public closure; boundary qualification uses the actual generated repository tree.
A generic schema compiler can retain the small historical
repository tree, but must not copy the legacy task snapshot or its large experiments.

## Checks and execution

Build with `python3 evaluator/verifiers/package.py --build .` from `build-input`;
this requires Requests and build-time networking. Assets are revision-pinned and
their SHA256 digests are saved. No model/dataset download occurs during execution.
From the compiled evaluator directory, run the declared self-test:
`python3 -I verifiers/prepare_launcher.py && python3 -m unittest discover -s verifiers -p 'test_*.py' -v`.
This adapts the compiler-generated `test.sh` and runs metric, process, snapshot and tensor tests;
tensor tests require the compiled PyTorch environment but no GPU or model download.
`python3 evaluator/verifiers/audit_assets.py` verifies every packaged row and hash.

Packaged protected execution starts through the adapted `tests/test.sh` with Linux root.
`evaluator/verifiers/launch.sh` remains an optional direct isolated bootstrap.
It immediately selects isolated Python and protects the evaluator directory. Public-safe
training assets are staged outside the protected directory. Training is unprivileged,
receives no evaluation inputs, and its process group is terminated before inference.
Only a bounded regular safetensors file crosses into a fresh trusted interpreter;
all config/tokenizer code stays trusted. Private gold never enters the training process
or the inference arguments. Candidate stdout/stderr tails explain failures.

Source snapshots precede paired baseline training, so edits during baseline execution
cannot change the candidate being tested. All required helper source is retained;
known environment, dataset and output directories are excluded. Snapshot time is charged.
The baseline is retrained, not loaded from a cached checkpoint. Baseline invalidity
is a construction error, never a candidate reward. Reward JSON contains only scalar
numbers; case metrics and timing details go to diagnostics JSON. The isolated shell
only bootstraps `verifiers/verify.py`, which owns orchestration and reward writing.
Trusted retrieval crashes and filesystem failures propagate as evaluator errors,
not invalid submissions. Invalid tensors have a distinct trusted exit status;
training failures and budget overruns remain submission invalidity. A shared trusted
preflight checks required packaged assets and usable CUDA devices before either
training process; unavailable infrastructure never becomes a candidate zero score.
Every valid full run uses exactly `(C-0.446)/(1-0.446)`, without clipping or flooring.
Freshly trained baseline B is diagnostic only; neither B nor any measurement changes
the anchor. Invalid submissions receive zero, while valid negative rewards survive.

`modal_qualification.py --build .` creates an owned, network-blocked 2-H200 sandbox,
tests the exact public command from its solver file view, compares control and three
training seeds per recipe, records hardware/logs, and terminates its sandbox in `finally`.
The qualification reference directory is root-only and is not part of the solver overlay.
Check `evidence` for actual observations; do not infer runtime or quality from this design.
The three standard six-hour solver trials require the external solver harness; they
are not replaceable by three random seeds or builder-authored experiments.
Use `--mode boundary --compiled-task PATH` to build from the actual generated
Dockerfile and solver tree and execute its adapted `tests/test.sh`. Public commands
run as an unprivileged user with no access to the evaluator or reference solution.

## Release gates

Do not mark this candidate released until the real OptimizationBuildPlan loader,
compiled public command, Linux boundary tests, repeated-seed reference improvement,
private paired execution and three six-hour solver trials have all completed.
The Taskforge schema loader/compiler is not importable in the construction Python.
The Werm CLI and historical generated task are available, but the standard solver-job
configuration is not supplied. Its dotenv auto-discovery also needs disabling in this
restricted workspace (`PYTHON_DOTENV_DISABLED=1`). JSON Schema validation and direct
Modal integration checks are not presented as substitutes for compiler qualification
or three independent solver trials. No ad hoc paid model calls were made.
Tiny's learning capacity and starter headroom must be established from actual runs.
Weights-only submission does not establish learning or prevent memorization.

## Measured public learning

This section and the subsequent attempts 1–4 are historical evidence under human-1.
Recorded rewards use `(1-B)/max(1-C,0.01)`, not the current fixed-anchor formula.
They are preserved, not recomputed or used to calibrate the new anchor.

The full network-blocked 2-H200 run in `evidence/1789000716` completed all seven
quality measurements plus the smoke check. Unadapted Tiny scored 0.187413 nDCG.
For seeds 1729, 2027 and 4093, starter nDCG was 0.447308, 0.446274 and 0.445282;
reference nDCG was 0.503865, 0.510353 and 0.511212. The paired gain averaged
0.062188 with sample standard deviation 0.004964; all three gains were positive.
Mean both-supports@2 rose from 0.091254 to 0.120009. Training took 117–122 seconds
and full retrieval 31–37 seconds for trained recipes, comfortably within 900/600.
These are public results, not evidence of successful independent solver trials.
Same-seed runs can differ slightly across GPU allocations; qualification uses repeated
seeds rather than claiming bitwise determinism. See the later boundary run for the
final snapshot-charged implementation and private paired check.

The final integration run is `evidence/1789001595`: all ten Linux tests, both cheap
public tools, the full paired public command and protected private launch passed.
Private baseline/reference nDCG was 0.461002/0.515294, giving reward 1.112011;
both-supports@2 was 0.099935/0.120768. Private training phases took 171.7 and 172.9
seconds; retrieval took 39.9 and 38.8 seconds. No private result guided a code change.
All four owned sandboxes, including two failed startup/debug attempts, are stopped.
`qualification_report.json` records remaining gates without claiming release readiness.

## Historical deterministic repair

Attempt 1 rejected the starter digest and the declared launcher's reward contract.
Public packaging now uses the compiler's `public_tests_dir` instead of a duplicate
164 MB overlay subtree. The same isolated launcher now owns numeric reward output;
there is no added wrapper, model, dataset, hardware substitution or scoring change.
Regression tests cover missing infrastructure, trusted child failure, malformed
tensors, the real shell launcher's numeric output and preservation of isolation.

The repair reruns CPU tests with the pinned tensor libraries, schema validation,
asset hashes/counts/joins and a materialized repository-plus-public-tools file view.
These checks do not certify the original digest mismatch as resolved: the actual
OptimizationBuildPlan loader/compiler remains unavailable in this sandbox. Nor do
they replace running the exact public commands in the compiled Linux solver image.
Earlier GPU evidence is historical evidence for unchanged learning/ranking behavior,
not a GPU qualification of the revised packaging and failure paths. No additional
GPU sandboxes, paid model calls or private-panel tuning were used for this repair.

The follow-up inexpensive checks pass 17 tests in 6.488 seconds. The new regression
copies only `public_tests` into a temporary solver workspace and runs the actual
public help and weights-check commands with the packaged pretrained Tiny. It does
not rely on sibling evaluator modules or fixtures. `evidence/repair-static-checks.json`
records schema, repository boundary, exact public file allowlist and dependency
closure checks; `evidence/asset-audit.json` records the repeated asset audit.
All four commands in `RETRIEVAL.md` were also exercised in the relocated real
repository plus overlay and public tools (`evidence/repair-public-view.json`).
The CPU weights check passed; the three GPU-dependent commands failed at the CUDA
preflight without training or emitting an invalid-submission score. Compiler access
was attempted again: the Taskforge module is absent from available Python environments
and the host project's Python environment is sandbox-blocked. The digest and compiler
contract gates therefore remain unconfirmed, not accepted or silently bypassed.
No training, retrieval, data, reward or hardware behavior changed in this follow-up.

## Entrypoint repair after attempt 2

Attempt 2 passed deterministic compilation; its review accepted the public closure,
repository boundary and reward mapping. That acceptance supersedes the historical
compiler uncertainty above. This repair does not reopen those resolved issues.

The private orchestration and numeric reward writer now live together in
`verifiers/verify.py`. `verifiers/launch.sh` only establishes isolated startup;
there is no output/result tuple contract or root-level Python entrypoint.
Standard unittest discovery replaces the root self-test wrapper, including in the
Linux qualification command. Infrastructure exceptions still clear stale rewards,
write diagnostics and propagate; invalid submissions still receive zero reward.

All 18 CPU tests pass in 6.237 seconds with no skips, including tensor validation,
launcher isolation, paired-path selection, numeric reporting and failure propagation.
Schema-v3, required paths, syntax, asset hashes/counts/joins and public-helper parity
also pass. The 22 public files and starter overlay match the accepted compiled tree
byte-for-byte. See `evidence/entrypoint-self-test.log` and
`evidence/entrypoint-static-checks.json`.

Every documented public command was exercised in a relocated copy of that compiled
solver file view: help and CPU weights validation pass; full, smoke and paired modes
stop at missing CUDA infrastructure before training, without a score. This macOS CPU
check is not a Linux GPU run (`evidence/entrypoint-public-view.json`). The Taskforge
module remains unavailable locally, so the revised private entrypoint still needs
pipeline recompilation and Linux qualification. No additional model calls, GPU
sandboxes, training, private tuning or release actions were performed.

## Attempt-3 contract repair and startup conflict

The plan now declares `verifiers/verify.py`, the existing substantive writer, rather
than its shell bootstrap. Numeric valid/reward output and failure classification
are unchanged. `launch.sh` now directly executes that file with `python -I`; the
embedded Python forwarding program is removed. The private entrypoint establishes
its trusted import path only after checking isolation and root. No root wrappers,
tuple-return reporting contract or duplicate public assets have been reintroduced.

Current evidence contradicts one implication of attempt 2's structural acceptance:
its retained `compiled/.../tests/test.sh` uses `python /tests/verifiers/launch.sh`
and exports a candidate-containing `PYTHONPATH`. It never executes the shell
bootstrap. Declaring the Python writer removes that language mismatch and exposes
the output contract, but the generated command still lacks `-I`. The schema has
no interpreter-flags setting. Re-executing after normal Python startup is not a
safe adaptation: candidate `sitecustomize.py` could already have executed. Keep
the isolation guard and use the packaged shell bootstrap (or `python -I` directly).
Unisolated compiled invocation is an infrastructure blocker, not a submission
failure. Do not claim that structural validation alone proves startup safety.
The accepted repository digest, public closure and task identity remain unchanged.

All 19 CPU tests pass in 6.850 seconds, with no tensor-test skips. Schema, syntax,
required paths, every asset hash/count/join, helper parity and the unchanged 22-file
public package pass. All documented public commands were rerun in a relocated
compiled solver file view; help and weights validation pass, while GPU modes stop
at CUDA preflight without training or a score. Evidence is in `evidence/contract-*`.
The OptimizationBuildPlan import still fails locally (`No module named 'taskforge'`).
No compiler or Linux GPU success is claimed, and no new GPU/model charges were
incurred. The candidate remains unreleased pending these infrastructure gates and
the previously required solver trials.

## Attempt-4 generated-launcher repair

Preserve attempt 4's accepted layout, contract discovery, public packaging and
repository boundary. The remaining startup mismatch is now handled during the
compiler's existing post-generation self-test step, not during verifier startup.
The declared command first runs `python -I verifiers/prepare_launcher.py`, then
standard unittest discovery. The adapter changes only the generated
`test.sh` invocation to `python -I /tests/verifiers/verify.py`. It accepts an already
adapted launcher and rejects unknown output. Keep this step and the modified
launcher in the packaged build; no compiler-source access or custom image is needed.
The substantive verifier, root/isolation guard, shell bootstrap, numeric writer,
models, data, training, retrieval and reward mapping are unchanged.

The exact declared command passes all 21 tests in 7.149 seconds, with no skips,
in a copy of the actual compiler-produced tree. New integration executes the
generated shell launcher rather than `launch.sh`, including its PYTHONPATH export.
Only absolute host paths, expensive runner work and the non-root host UID are
adapted in the temporary test fixture. The real writer emits numeric valid/reward
for valid and invalid cases; infrastructure failure clears a stale reward and
propagates diagnostics. Candidate startup/shadow modules never execute.
See `evidence/startup-self-test.log` and `evidence/startup-static-checks.json`.

Schema, syntax, required paths, asset hashes/counts/joins and public-helper parity
pass. All 22 public files and the overlay remain byte-identical to the accepted
compiled tree. All documented public commands were rerun in its relocated solver
view: help and weights checking pass; GPU modes stop at unavailable CUDA without
training or a score (`evidence/startup-public-view.json`). No GPU runs, model calls,
private tuning, sibling changes or releases occurred. The full compiler module
is still unavailable locally; the accepted structural result is not reopened.
Final Linux/GPU qualification and the three solver trials remain pending.

## Attempt-6 repair under human-2

Keep the accepted repository boundary, public packaging, verifiers/utils layout,
safe tensor handoff and generated-launcher adaptation. The declared self-test now
uses `python3` for both steps, so the host need not provide a `python` alias.
Its exact command passes 22 tests without skips in the updated compiled-tree copy,
using a PATH with `python3` but no `python`. No checks were bypassed. A newly exposed
test-fixture path collision (temporary paths themselves contained `/tests`) is fixed
by one-pass relocation, without changing the real launcher or weakening isolation.

Human-2 explicitly supersedes the accepted historical remaining-error reward.
The only scoring implementation now takes C alone and returns `(C-0.446)/(1-0.446)`.
Both full public modes and the private paired path share it; baseline B remains a
freshly trained diagnostic. Tests cover endpoints, valid negatives, baseline
independence, the invalid-zero gate and infrastructure exceptions. Historical
measurements above and in `qualification_report.json` retain their old definition.
Models, training recipes, tokenizer/config, ranking, data and budgets are unchanged.

Schema-v3, required paths, the 22-file public allowlist, shared-helper parity and
all asset hashes/counts/joins pass (`evidence/attempt6-*`). The actual compiler
module remains unavailable locally; accepted structural findings are not reopened.
For packaged qualification, a copy of the supplied generated task receives only
the new overlay/public closure and evaluator changes, then the declared self-test.
Its original Dockerfile builds successfully; the adapted generated launcher, not
the optional shell bootstrap, is selected for private scoring.

The first Modal launch (`evidence/1789014600`) failed before any workload request:
its command-router TLS handshake could not use the host's system CA bundle.
The owned sandbox was stopped; no submission or score was produced. Retrying with
`SSL_CERT_FILE` set to the installed Certifi bundle preserves certificate verification
and allows execution (`evidence/1789015668`). This is host tooling configuration,
not a change to task networking, CUDA resources or candidate validity.

The retry passes all 22 Linux tests and every documented public command on two
measured H200s. The actual generated `/tests/test.sh` also passes private paired
evaluation. Fresh public/private reference nDCG is 0.503666/0.515868, with new-formula
rewards 0.104091/0.126115. Starter-only public reward is 0.002362, not one. Training
takes 120.59–124.20 seconds and retrieval 31.62–31.97 seconds. Full metrics, slices,
times and hardware are in `evidence/1789015668/summary.json`. Both new sandboxes
are stopped; no extra model calls or private tuning occurred.

This is not release readiness. The standard compiler must recompile the updated
draft, and the standard three one-hour solver trials still require their agent/model
job configuration. Werm's default oracle is not a substitute. Also assess baseline
strength over the full allowance: this unchanged starter uses only about 121 of
900 training seconds. Historical repeated-seed gains do not settle solver difficulty.
