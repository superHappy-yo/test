# SCIC-v2 artifact

This archive is separate from `jsa_supplement` and `jsa_ablation`. Only completed,
independent primary GPU workers can contribute to acceptance statistics. Never
initialize TorchProfiler/CUPTI in a primary worker. `exclusions.json` identifies
pilot/debugging data; failed gates and engine initialization logs remain archived.

Environment: PyTorch 2.13.0+cu126, Triton 3.7.1, Transformers 5.17.0, vLLM 0.31.0;
A40 FP16 TP1, CPU threads one. Pretrained Qwen/Qwen3-0.6B is pinned to
`c1899de289a04d12100db370d81485cdf75e47ca`; provenance records file SHA256 values.
Checkpoints are local in `.experiments/models`, not copied into this artifact.

Operator protocol: 32 distinct projection geometries, 23 training / nine held-out
square-2048/8192 geometries; fixed vs six rotated weights; direct vs static CUDA
Graph. Each validated candidate: 50 warmups, 200 event observations and five
synchronized host batches of 40 calls. Selection uses fresh probes: three
warmups, five host batches of eight calls. Every numerical audit includes three
fixed input seeds, unit/nonunit norm weights, and stress inputs; both output and
residual use atol=rtol=1e-3 and relative L2<=1e-3. Candidate rejection is retained.
`max-autotune` retains its default numerical optimization and must pass the same
component gate. It is not silently reclassified as valid when casts differ.

`selections.json` contains a **legacy band transferred to the five-candidate
pool**, labeled `legacy`; this diagnostic is not the actual original SCIC
algorithm. The separate `operator-legacy` workers measure unmodified
`ProjectionSite('scic')`, including its exact-key dispatcher and two-candidate
GPU-event calibration. Only this separate stage supports an old-SCIC comparison. It also evaluates a
semantic-preserving max-autotune control with `emulate_precision_casts=True`,
rather than inferring its speed from a numerically rejected default-max path.

The full HF model checks every prefill/64 teacher-forced decode output against
the optimized original using atol=rtol=1e-2 and relative L2<=1e-3. If the strict
relative-L2 gate fails but the elementwise gate passes, timing is retained as
**exploratory** (`numerics_passed=false`); it cannot support acceptance or system
acceleration claims. Numerical diagnostics also compare eager and compiled
originals. No threshold is adjusted in response to observed performance.

vLLM uses actual greedy generation: six random-model workloads, four pretrained
workloads, 128 requests and 64 tokens/request. Token IDs/hashes, TTFT, ITL,
throughput and preparation reports are retained. The dynamic token dimension
requires one frozen candidate per layer; its preparation mixture is 64 decode
calls at M=8 per one prefill at M=2048. This deployment constraint is distinct
from per-exact-shape operator selection. Mutation is declared on residual; no
extra steady-state residual clone is introduced by the new serving operator.

Acceptance is held-out steady speed >=98% of full calibration **and total
preparation** reduction >=30%. Probe-only reductions are insufficient; operator
rows record common exhaustive candidate compilation/validation so total costs
must include them. AI contribution is compared independently against no-AI.
System speed claims require >=1.02x and process-cluster bootstrap 95% lower
bound >1, plus strict numerical validity. Cluster at GPU/run, never bootstrap
individual timer observations. Use 5000 cluster resamples; keep execution and
reuse domains separate.

Reproduction:

```sh
.experiments/envs/vllm/bin/python paper/jsa/scripts/run_jsa_v2.py --stage operator
.experiments/envs/vllm/bin/python paper/jsa/scripts/run_jsa_v2.py --stage operator-legacy
.experiments/envs/vllm/bin/python paper/jsa/scripts/run_jsa_v2.py --stage operator-vllm
.experiments/envs/vllm/bin/python paper/jsa/scripts/run_jsa_v2.py --stage hf --prior paper/data/jsa_v2/gpu0_run02/operator/priors.json
.experiments/envs/vllm/bin/python paper/jsa/scripts/run_jsa_v2.py --stage hf-pretrained --prior paper/data/jsa_v2/gpu0_run02/operator/priors.json
.experiments/envs/vllm/bin/python paper/jsa/scripts/run_jsa_v2.py --stage serving --prior paper/data/jsa_v2/gpu1_run00/operator-vllm/priors.json
.experiments/envs/vllm/bin/python paper/jsa/scripts/run_jsa_v2.py --stage serving-pretrained --prior paper/data/jsa_v2/gpu1_run00/operator-vllm/priors.json
```

All workers run serially after two idle GPU observations. Source snapshots and
per-worker hashes allow revisions made after pilots to be distinguished.

Resource change: after the balanced HF operator series, GPU 0 became occupied by another process. The remaining primary random-model domains use five independent processes on physical GPU 1; pretrained domains use three. The completed GPU-0 serving-semantic operator run is archived separately and excluded conservatively. See `provenance/resource_change.json`.

Control correction: the initial legacy-control workers omitted custom-operator
registration and produced no admitted timing rows. They remain in
`operator-legacy_registration_failure/`; corrected workers explicitly register
the operator and use a cast-preserving max-autotune reference. Only corrected
workers paired with the same primary GPU/run support baseline comparisons.
See `provenance/control_correction.json`.

Preparation accounting: the exhaustive operator audit prepares all candidates.
Its total-cost ratios describe that scan protocol, not cold lazy preparation.
`summary.json/preparation` separately summarizes actual full-model preparation
(compilation, validation, probing) against full calibration. These measurements
retain their observed compiler-cache state and numerical validity; invalid
logits cannot be converted into a system acceleration claim.

`prior_audit.json` describes training-only geometry support and empirical LOOG
uncertainty. It does not fit new hyperparameters or features to held-out results.

Serving correctness fallback: when the candidate pool or joint decode/prefill
validation has no admitted implementation, preparation restores that layer's
original vLLM projection and forward method before tracing/capture. Records
mark `deployed=native` and `deployment_fallback`; native layers do not establish
a fused-kernel gain. This revision precedes all primary serving workers.
