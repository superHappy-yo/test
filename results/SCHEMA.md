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
| `bench_gemm.csv` | vendor matmul latency. `achieved_ai` is empty unless Nsight JSON was passed. `proxy_ai` is torch.profiler allocator memory when `--profile torch` |
| `decision_log.csv` | analytical dual-plan decision for each bench shape, plus byte-derived regime when a byte AI exists |
| `agreement.json` | agreement rates. `quote_as_scic_accuracy` is true only for status `NSIGHT_DRAM` |
| `fusion_proxy.csv` | present when `--fusion-proxy` is set. Read `honesty`. Not MemMode or ComputeMode |
| `gpu_measurements.json` | device metadata, agreement summary, fusion-proxy summary, explicit nulls |
| `summary.json` | machine, crossings, highlights, TBD list |
| `fig_*.png` | analytical figures, plus `fig_vendor_matmul_tflops.png` when a bench ran |
| `nsight_dram.json` | optional, from `scripts/nsight_dram.sh`. Not written by the CLI itself |
| `SEND_BACK.txt` | checklist of what to return |

## Fields that stay empty until a human measures them

- `achieved_ai`, until a Nsight JSON is ingested (`--nsight-json` or `TRY_NSIGHT=1`)
- `proxy_ai` is allowed to be filled by `--profile torch` and must stay labeled `TORCH_PROFILER_MEMORY_NOT_DRAM`
- `measured_hbm_bandwidth_bytes_per_s`, `measured_ridge_flops_per_byte`
- `measured_winner` in `dual_plans.csv`
- MemMode versus ComputeMode kernel time (no such kernel is emitted; `fusion_proxy.csv` does not count)
- end-to-end tokens/s
- `quote_as_scic_accuracy` stays false for profiler memory and for `NSIGHT_DRAM_PROCESS_SUM`

Confirm profiler metric names on the target driver (`ncu --query-metrics`) before filling DRAM bytes. Achieved AI is FLOPs divided by measured DRAM bytes, not FLOPs divided by latency. `--machine a40` selects the A40 datasheet preset (149.7 TFLOP/s, 696 GB/s). That preset is not a measured ridge. The archived A40 matmul sweep used `--machine H100_SXM_FP16`, so its regime column is the H100 datasheet.
