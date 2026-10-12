# A40 wall-clock bakeoff (selector before the attention fix)

This directory is the upload from `scripts/run_baseline_bakeoff.sh` on the branch recorded in `RUN_NOTES.txt` (`cursor/docker-baseline-bakeoff-d966`, commit `81ec1c5`). The files are copied as returned. They are not regenerated, and the numbers are not edited.

- `gpu_measurement_status`: `BAKEOFF_WALLCLOCK` (`bakeoff_summary.json`)
- `nsight_used`: false
- `quote_as_scic_accuracy`: false
- `graph_inspected`: false
- package version in the summary: `0.3.0`
- 77 latency rows, all `ok=true`; 12 decisions; `selected_baseline` counts `always_fuse` 6 and `always_unfuse` 6

`scic_selected` copies `timing_copied_from`. It does not include plan-switch overhead.

This run used the pre-fix attention rule: analytical ComputeMode selected `always_unfuse`. That misses `prefill_s512` and `prefill_s2048`, where `always_fuse` is faster. The code change that scores `compile_sdpa_chain` (timed as `always_fuse`) for library SDPA is **not** what these rows measured. Please re-run the bakeoff on an idle GPU before quoting the fixed selector.

`RUN_NOTES.txt` records a busy GPU (83% and 77% utilization at script start) and Dynamo `cache_size_limit=8`. `nvidia_smi.txt` is the snapshot saved with the upload (GPU 0 at 81%, GPU 1 at 78%).
