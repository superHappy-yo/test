# A40 wall-clock bakeoff, second run (after the attention fix)

This directory is the upload from `scripts/run_baseline_bakeoff.sh` on the branch recorded in `RUN_NOTES.txt` (`cursor/bakeoff-numbers-attention-fix-0187`, commit `cb3992f`). The files are copied as returned. They are not regenerated, and the numbers are not edited.

- `gpu_measurement_status`: `BAKEOFF_WALLCLOCK` (`bakeoff_summary.json`)
- `nsight_used`: false
- `quote_as_scic_accuracy`: false
- `graph_inspected`: false
- package version in the summary: `0.3.0`
- 77 latency rows, all `ok=true`; 12 decisions; `selected_baseline` counts `always_fuse` 8 and `always_unfuse` 4

`scic_selected` copies `timing_copied_from`. It does not include plan-switch overhead.

This run used the library-SDPA epilogue policy `compile_sdpa_chain`, timed as `always_fuse`. All five attention decisions copy `always_fuse`. GEMM is unchanged: ComputeMode and the compute side of the ridge still copy `always_unfuse`. On this CSV that GEMM choice leaves a faster `torch.compile` path on the table (`ridge_m256`, `ridge_m512`, `prefill_m2048`, `prefill_m4096`). This directory predates the wall-clock GEMM probe. Do not relabel those `always_unfuse` cells. A later bakeoff fills `measured_winner` from a few-shot wall-clock on the machine that runs it. That probe is not DRAM AI and not SCIC accuracy. A busy GPU can flip it.

`bakeoff_summary.json` still contains the generator note that asks for a re-bakeoff and names `paper/data/bakeoff` as the pre-fix selector. That sentence was emitted by the code that wrote the file. The CSVs in this directory are the post-fix wall-clock. `paper/data/bakeoff/` remains the pre-fix archive.

This was not an idle-GPU measurement. `gpu_selection.json` has `"gpu": 1` and `"idle_found": false`. Six pre-run samples did not find an idle device. `RUN_NOTES.txt` records GPU 1 at 92% utilization with 13490 MiB allocated by other workloads when the script started. This upload did not include `nvidia_smi.txt`, `torch_version.txt`, or `run.log`.

`RUN_NOTES.txt` also records Dynamo `cache_size_limit=64` and no cache-limit warning in `run.log`. Some max-autotune candidates exceeded A40 shared-memory limits and were rejected. Every final CSV row is `ok=true`. Graphs were not inspected. Nsight was not run.
