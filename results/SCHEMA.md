# `results/` schema

The CLI creates this directory. Generated CSV, JSON, and PNG files are gitignored. GPU fields that this repo cannot fill are JSON `null` or the token `TBD_NSIGHT`.

`gpu_measurement_status` in `summary.json` is one of:

| status | meaning |
| --- | --- |
| `PLACEHOLDER_NOT_RUN` | tables only; no timer |
| `PLACEHOLDER_NO_GPU` | timer ran on CPU or NumPy; not an H100 result |
| `PARTIAL_VENDOR_MATMUL` | CUDA `torch.matmul` latencies are real; MemMode and ComputeMode kernels were not timed. A fusion proxy, if present, is a separate file and does not change this status |

## Files

| file | contents |
| --- | --- |
| `gemm_shapes.csv` | analytical FP16/FP32 AI and regime for the fixed shape list |
| `gemm_M_sweep.csv` | analytical AI versus M for N=K=4096 |
| `gemm_M_sweep_cpu.csv` | optional NumPy proxy (`--cpu-proxy`) |
| `attention_intensity.csv` | prefill QK^T, softmax, FA-style IO, unfused scores |
| `attention_decode.csv` | MHA / GQA decode AI |
| `rmsnorm_epilogue.csv` | byte-save fraction and roofline speedup; launch column is assumed |
| `ew_chains.csv` | elementwise chain AI |
| `moe_grouped.csv` | ragged grouped-GEMM AI and per-expert regimes |
| `quant_epilogue.csv` | mixed-dtype byte what-if |
| `dual_plans.csv` | MemMode / ComputeMode decisions; `kernel_emitted` is false |
| `crossings.json` | M at 0.5 R, R, and 2 R |
| `bench_gemm.csv` | vendor matmul latency. `achieved_ai` is empty unless Nsight JSON was passed. `proxy_ai` is allocator memory when `--profile torch`: profiler events (`TORCH_PROFILER_MEMORY_NOT_DRAM`) or, if those events are empty, a caching-allocator delta (`TORCH_ALLOCATOR_DELTA_NOT_DRAM`). Neither is DRAM |
| `decision_log.csv` | analytical dual-plan decision for each bench shape, plus byte-derived regime when a byte AI exists |
| `agreement.json` | agreement rates. `quote_as_scic_accuracy` is true only for status `NSIGHT_DRAM` |
| `fusion_proxy.csv` | present when `--fusion-proxy` is set. Read `honesty`. Not MemMode or ComputeMode |
| `gpu_measurements.json` | device metadata, agreement summary, fusion-proxy summary, explicit nulls |
| `summary.json` | machine, crossings, highlights, TBD list |
| `fig_*.png` | analytical figures, plus `fig_vendor_matmul_tflops.png` when a bench ran |
| `nsight_dram.json` | optional, from `scripts/nsight_dram.sh`. Not written by the CLI itself. Strongly recommended via `TRY_NSIGHT=1`. If `ncu` is missing or fails (`ERR_NVGPUCTRPERM` included), `scripts/run_a40.sh` writes `nsight_skipped.txt`. See `docs/NSIGHT_PERMISSIONS.md` |
| `bandwidth_estimate.json` | optional copy-kernel estimate from `scripts/bandwidth_probe.sh`. Status `ESTIMATE_COPY_NOT_NSIGHT`. Not a measured ridge |
| `SEND_BACK.txt` | checklist of what to return |
| `bakeoff_latency.csv` | from `python -m shapefuse bakeoff` or `scripts/run_baseline_bakeoff.sh`. Columns include `baseline`, `shape`, `median_ms`, `speedup_vs_eager`, `honesty`. No Nsight. `quote_as_scic_accuracy` is false. `sdpa` is not FlashAttention-3. `scic_selected` copies another row's median |
| `bakeoff_decisions.csv` | SCIC regime, MemMode or ComputeMode, and `selected_baseline` (`always_fuse` or `always_unfuse`) |
| `bakeoff_summary.json` | `BAKEOFF_WALLCLOCK` on CUDA, `PLACEHOLDER_NO_GPU` otherwise. `nsight_used` is false |

## Fields that stay empty until a human measures them

- `achieved_ai`, until a Nsight JSON is ingested (`--nsight-json` or `TRY_NSIGHT=1`)
- `proxy_ai` is allowed to be filled by `--profile torch`. Profiler events stay labeled `TORCH_PROFILER_MEMORY_NOT_DRAM`. A caching-allocator fallback is `TORCH_ALLOCATOR_DELTA_NOT_DRAM`. `TORCH_PROFILER_EMPTY` means neither source produced bytes. None of these is DRAM
- `measured_hbm_bandwidth_bytes_per_s`, `measured_ridge_flops_per_byte` (a copy-kernel estimate does not fill these; see `bandwidth_estimate.json`)
- `measured_winner` in `dual_plans.csv`
- MemMode versus ComputeMode kernel time (no such kernel is emitted; `fusion_proxy.csv` does not count)
- end-to-end tokens/s
- `quote_as_scic_accuracy` stays false for profiler memory, for `NSIGHT_DRAM_PROCESS_SUM`, and for every bakeoff row
- bakeoff medians, until `scripts/run_baseline_bakeoff.sh` has actually run on CUDA. Do not invent them. A CPU smoke run is `PLACEHOLDER_NO_GPU`

Confirm profiler metric names on the target driver (`ncu --query-metrics`) before filling DRAM bytes. `ncu` is Nsight Compute, separate from the driver; if it is missing, or if it fails, `scripts/run_a40.sh` writes `nsight_skipped.txt` when `TRY_NSIGHT=1` and leaves `achieved_ai` empty. `ERR_NVGPUCTRPERM` is a performance-counter permission error: see `docs/NSIGHT_PERMISSIONS.md`, reboot, then `TRY_NSIGHT=1` again. Achieved AI is FLOPs divided by measured DRAM bytes, not FLOPs divided by latency. `--machine a40` selects the A40 datasheet preset (149.7 TFLOP/s, 696 GB/s). That preset is not a measured ridge. The archived third A40 package used `--machine A40_FP16`. Its regime column is that datasheet. `ncu` failed with `ERR_NVGPUCTRPERM`, `proxy_ai` is `TORCH_PROFILER_MEMORY_NOT_DRAM`, and `quote_as_scic_accuracy` is false. `bandwidth_estimate.json` in that package is `ESTIMATE_COPY_NOT_NSIGHT` and does not fill `measured_ridge_flops_per_byte`.
