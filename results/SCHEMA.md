# `results/` schema

The CLI creates this directory. Generated CSV, JSON, and PNG files are gitignored. GPU fields that this repo cannot fill are JSON `null` or the token `TBD_NSIGHT`.

`gpu_measurement_status` in `summary.json` is one of:

| status | meaning |
| --- | --- |
| `PLACEHOLDER_NOT_RUN` | tables only; no timer |
| `PLACEHOLDER_NO_GPU` | timer ran on CPU or NumPy; not an H100 result |
| `PARTIAL_VENDOR_MATMUL` | CUDA `torch.matmul` latencies are real; fusion kernels were not timed |

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
| `bench_gemm.csv` | vendor matmul latency; `achieved_ai` empty |
| `gpu_measurements.json` | device metadata and explicit nulls |
| `summary.json` | machine, crossings, highlights, TBD list |
| `fig_*.png` | analytical figures |
| `SEND_BACK.txt` | checklist of what to return |

## Fields that stay empty until a human measures them

- `nsight_dram_bytes`, `achieved_ai`
- `measured_hbm_bandwidth_bytes_per_s`, `measured_ridge_flops_per_byte`
- `measured_winner` in `dual_plans.csv`
- MemMode versus ComputeMode kernel time (no kernel is emitted)
- end-to-end tokens/s

Confirm profiler metric names on the target driver (`ncu --query-metrics`) before filling DRAM bytes. Achieved AI is FLOPs divided by measured DRAM bytes, not FLOPs divided by latency.
