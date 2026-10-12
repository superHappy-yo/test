# A40 wall-clock bakeoff, fourth run (`CALIBRATE=all-gemm`, idle GPU)

This directory is the upload from `scripts/run_baseline_bakeoff.sh` on the branch recorded in `RUN_NOTES.txt` (`cursor/calibrated-scic-gemm-f3e1`, commit `361366c`). The files are copied as returned. They are not regenerated, and the numbers are not edited.

- `gpu_measurement_status`: `BAKEOFF_WALLCLOCK` (`bakeoff_summary.json`)
- `nsight_used`: false
- `quote_as_scic_accuracy`: false
- `graph_inspected`: false
- package version in the summary: `0.3.1`
- `gemm_calibration.scope`: `all-gemm` (seven GEMM shapes probed; attention is not probed)
- 77 latency rows, all `ok=true`; 12 decisions; `selected_baseline` counts `always_fuse` 8 and `always_unfuse` 4
- `selection_source` counts: `measured_wallclock` 7, `library_sdpa` 5

`scic_selected` copies `timing_copied_from`. It does not include plan-switch overhead.

This run uses the library-SDPA epilogue policy `compile_sdpa_chain`, timed as `always_fuse`. All five attention decisions copy `always_fuse` with `selection_source=library_sdpa`. GEMM decisions follow a few-shot wall-clock probe (`measured_wallclock`): decode M ∈ {1, 32}, ridge M ∈ {128, 256} copy `always_unfuse`; ridge M = 512 and prefill M ∈ {2048, 4096} copy `always_fuse`. That probe is not DRAM AI and not SCIC accuracy. GPU contention can flip the winner.

`paper/data/bakeoff/` remains the pre-fix attention selector. `paper/data/bakeoff_v2/` is the post-fix wall-clock on a busy GPU and predates GEMM calibration. This directory is the calibrated GEMM run on a waited-for idle physical GPU (see `RUN_NOTES.txt`).

This upload did not include `gpu_selection.json`, `gpu_monitor.csv`, `nvidia_smi.txt`, `torch_version.txt`, or `run.log`.

`RUN_NOTES.txt` records Dynamo `cache_size_limit=64` and no cache-limit-hit warning. Graphs were not inspected. Nsight was not run.
