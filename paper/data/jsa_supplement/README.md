# Executed supplementary evaluation

The follow-up adapter diagnosis in `../jsa_ablation/` isolates profiler sessions
from final timing processes. Its PyTorch 2.5.1 intervention detects persistent
post-profiler timing changes. This earlier operator archive inspected kernels
before candidate timing in the same process; its operator observations retain
that instrumentation context and are not pooled with the new unprofiled
matrix. Earlier HF model and vLLM protocols are reported separately.

These are complete-function measurements, not copied baseline latencies. Failed
and unsupported candidates stay in `latency.csv`; only `execution_status=ok`
rows enter performance summaries. Smoke directories and interrupted processes
are excluded. No pretrained checkpoint is downloaded.

## Experiments

| Suite | Environment and purpose |
| --- | --- |
| `torch251_operators*` | PyTorch 2.5.1+cu121; 21 GEMM and 9 attention shapes, real dispatch, static compiler/library baselines |
| `torch214_operators*` | Independently installed PyTorch 2.14.1+cu126 with matching baselines |
| `redfuser_shipped_gpu0` | Committed RedFuser attention kernel with literal dimensions/grid adapted, plus compiled FP32 RMSNorm |
| `neptune_torch_gpu1` | Official Torch-callable Triton bridge with launch integer type compatibility patch; five prefill/causal shapes |
| `torch214_compiler_graphs_gpu1` | Framework-managed graphs with ordinary unmarked tensor arguments; input staging copies are observed |
| `vllm_gpu1` | Actual vLLM 0.31.0 V1 engine, fixed random weights, four policies and six serving workloads |
| `neptune_gpu1` | Official Neptune schedules, PyTorch 2.6.0+cu126/Triton 3.2, plus compiled FP32 RMSNorm |
| `torch251_system_gpu*` | Complete 83,900,160-parameter random-weight Qwen3, original/fixed-native/fixed-compiled/SCIC |

Each operator process records three input audits, kernel/backend traces when
available, 50 warmups, 200 CUDA-event samples, and a separate synchronized host
batch. `none` and `manual` CUDA Graph groups are separate. SCIC probes use three
warmups and ten event samples per candidate, plus the helper's separate ten-call
host batch. `calibration_ms` includes those calls and synchronization. `setup_ms`
includes construction, compilation, and numerical auditing; it is not an
isolated compiler-duration measurement. Prepared cache signatures include exact
shape/stride/dtype/device/implementation/device identity. Every selector row is
independently executed and timed.

The first PyTorch 2.5.1 GPU-0 series began with fixed candidate order; later
series use seeded shuffle. Source hashes and per-candidate order fields identify
the protocol. FlashAttention-2 was added in the GPU-1 environment; its coverage
must be read from the admitted rows rather than assumed to equal all suites.
CUDA-event intervals may include host launch gaps. They are not active-kernel
time. Kernel names do not establish DRAM bytes or Roofline classification
accuracy.

## Reproduction

Use isolated virtual environments. The project keeps these GPU dependencies
optional. Install the requested official PyTorch CUDA wheel, and use
Transformers 4.51.3 for the small-model PyTorch suite. Ampere-compatible
FlashAttention-2 is optional and must be installed against that environment's
PyTorch ABI. No global Python packages or driver were upgraded.

```bash
python paper/jsa/scripts/run_jsa_supplement.py --python /path/to/env/bin/python \
  --stage operator --gpus 0,1 --runs 3 --graphs none,manual --out /new/operator-results
python paper/jsa/scripts/run_jsa_supplement.py --python /path/to/hf-env/bin/python \
  --stage system --gpus 0,1 --runs 3 --graphs none --timeout 600 --out /new/system-results
```

Outputs must be new directories. `--resume` continues an interrupted operator
or PyTorch-model suite without overwriting completed processes. Incomplete
directories are retained with an `.incomplete_*` suffix. `--batch-cases` groups the complete operator shape matrix into one process per device/run/graph, retaining per-shape audits and paired analysis. The PyTorch 2.14.1 GPU-1 series uses this option; GPU-0 uses isolated case processes. The runner checks idle
GPUs and kills only its own process group on timeout.

Recent source pins: RedFuser `56bf0b28c569fe160fd0a53a87a9d7f113852940`,
Neptune `f79750113db136e96516c4bb86a980fb1249559c`. RedFuser's generator uses a
separate child process/runtime; `redfuser_compatibility.patch` records its three
compatibility changes. Regeneration still failed the numerical gate and has no
speedup result. `redfuser_shipped` is a separate artifact, not successful
regeneration. It supports only the five unmasked MHA cases with D=64. Neptune's
plain-attention adapter supports seven cases; GQA is explicitly unsupported. The legacy TVM/Triton bridge crashed under whole-chain graph capture, so direct calls and unsupported captures are reported separately.

Build the pinned official TVM sources with CUDA enabled. Neptune additionally
uses LLVM 15.0.7; optional libbacktrace was disabled. Set `NEPTUNE_ROOT`, its
Python source paths, and LLVM library path in that environment. RedFuser sets
`REDFUSER_ROOT` and `REDFUSER_GENERATOR_PATH`, with TileLang in the parent and the
generator's matching TVM FFI in the child. The two FFI runtimes are never loaded
into the same Python process. See `shapefuse/research.py` for the adapters.

```bash
python paper/jsa/scripts/run_jsa_supplement.py --python /path/to/research-env/bin/python \
  --stage research --gpus 0 --runs 3 --graphs none,manual \
  --methods native,compile_default,compile_max_no_graph,redfuser_shipped \
  --out /new/research-results
```

For vLLM use its own environment and process per policy. The engine uses fixed
random dummy weights, token-ID inputs, 64 output tokens, prefix caching, chunked
prefill, and the framework's compilation/graph defaults. A reduced `--requests`
count is for smoke checks only, and is recorded.

```bash
python paper/jsa/scripts/run_jsa_vllm.py --python /path/to/vllm-env/bin/python \
  --gpus 0 --runs 3 --requests 128 --out /new/vllm-results
```

## Analysis

```bash
python paper/jsa/scripts/analyze_jsa_supplement.py --out paper/data/jsa_supplement/analysis
python paper/jsa/scripts/plot_jsa_supplement.py --summary paper/data/jsa_supplement/analysis/summary.json
```

Check `operator_summary.csv`, common-paired sample counts, `candidate_status.csv`,
`calibration_summary.csv`, and `system_summary.csv`. Speed ratios below one
are slowdowns. Confidence intervals resample process/device repetitions rather
than treating timing iterations as independent experiments. A measured pool
minimum is not a global kernel optimum. Random-weight profiling does not
evaluate language quality or pretrained-model task accuracy.

Neptune Torch-launch compatibility uses `neptune_torch_metadata.patch` to convert TVM integer launch values to equal Python integers. Original failures are retained in `diagnostic_neptune_torch_initial/`. This bridge supports five prefill/causal shapes; multi-kernel decode composition and GQA are unsupported. Its successful rerun uses a separate Triton cache.

Environment package locks are in `environment_locks/`. Large temporary wheel caches and the completed PyTorch 2.14 environment were removed after measurements to reduce staged disk use; `cleaned_wheel_hashes.json` retains their hashes. Source, patches and raw measurements remain.

Serving analysis additionally runs `python paper/jsa/scripts/analyze_jsa_vllm.py`. It compares every request token hash against the matching original engine before admitting a performance pair. Batch throughput includes deterministic prompt construction; per-request TTFT starts after construction. Client IPC, scheduling, KV cache, sampling and streaming are included; tokenizer and HTTP transport are excluded. P95 values in summaries are medians of per-process percentiles.
