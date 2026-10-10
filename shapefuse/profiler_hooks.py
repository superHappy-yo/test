"""Optional byte estimates for the matmul sweep.

Two sources, with different honesty:

- Nsight Compute DRAM metrics, loaded from JSON produced by
  ``scripts/nsight_dram.sh``. Those are device DRAM counters. A process-wide
  sum can still include copies, so the status string says so.
- ``torch.profiler`` memory. That is allocator traffic, not DRAM bytes. It
  must not be stored in ``achieved_ai``.

Analytical cold-cache bytes are not a measurement and are not returned here.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

NSIGHT_DRAM = "NSIGHT_DRAM"
NSIGHT_PROCESS_SUM = "NSIGHT_DRAM_PROCESS_SUM"
TORCH_MEMORY = "TORCH_PROFILER_MEMORY_NOT_DRAM"
TORCH_EMPTY = "TORCH_PROFILER_EMPTY"
TORCH_UNAVAILABLE = "TORCH_PROFILER_UNAVAILABLE"
PROFILER_OFF = "PROFILER_OFF"
TBD_NSIGHT = "TBD_NSIGHT"

# Isolated kernel DRAM. A process sum is real DRAM but not this.
STRICT_NSIGHT_STATUSES = frozenset({NSIGHT_DRAM})
# Enough to fill achieved_ai. Not enough to quote SCIC accuracy.
NSIGHT_AI_STATUSES = frozenset({NSIGHT_DRAM, NSIGHT_PROCESS_SUM})


def load_nsight_json(path: Path) -> dict[str, dict]:
    """Map bench label -> row dict with ``dram_bytes`` and ``status``."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    default_status = str(payload.get("status") or NSIGHT_PROCESS_SUM)
    if default_status not in NSIGHT_AI_STATUSES:
        raise ValueError(
            f"nsight JSON status must be one of {sorted(NSIGHT_AI_STATUSES)}, got {default_status!r}"
        )
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("nsight JSON needs a non-empty 'rows' list")
    out: dict[str, dict] = {}
    for row in rows:
        label = str(row["label"])
        nbytes = float(row["dram_bytes"])
        if nbytes <= 0:
            raise ValueError(f"dram_bytes for {label} must be positive, got {nbytes}")
        status = str(row.get("status") or default_status)
        if status not in NSIGHT_AI_STATUSES:
            raise ValueError(f"bad per-row nsight status {status!r} on {label}")
        out[label] = {"dram_bytes": nbytes, "status": status}
    return out


def parse_ncu_csv(text: str, metrics: list[str]) -> dict[str, float]:
    """Pull metric values out of an ``ncu --csv`` report.

    Supports a long table with ``Metric Name`` and ``Metric Value`` columns.
    Lines that are not CSV data (ncu banners) are skipped. Returns only the
    requested metrics that were found. Units are whatever ncu printed; the
    capture script asks for ``*.sum`` byte counters.
    """
    wanted = {name.strip() for name in metrics if name.strip()}
    found: dict[str, float] = {}
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("=="):
            continue
        if stripped.startswith('"') or "," in stripped:
            lines.append(stripped)
    if not lines:
        return found
    reader = csv.DictReader(lines)
    if reader.fieldnames is None:
        return found
    fields = {name.strip().strip('"'): name for name in reader.fieldnames if name}
    name_key = _field(fields, ("Metric Name", "metric_name", "Name"))
    value_key = _field(fields, ("Metric Value", "metric_value", "Value"))
    if name_key and value_key:
        for row in reader:
            metric = (row.get(name_key) or "").strip().strip('"')
            if metric not in wanted:
                continue
            parsed = _float_or_none(row.get(value_key))
            if parsed is not None:
                found[metric] = parsed
        return found
    # Wide layout: one column per metric.
    for metric in wanted:
        key = fields.get(metric)
        if key is None:
            continue
        for row in csv.DictReader(lines):
            parsed = _float_or_none(row.get(key))
            if parsed is not None:
                found[metric] = parsed
    return found


def profile_matmul_memory(m: int, n: int, k: int, cuda: bool) -> dict:
    """One matmul under torch.profiler.

    ``bytes`` is allocator memory (CUDA or CPU), not DRAM traffic. ``status``
    is never an Nsight status.
    """
    try:
        import torch
        from torch.profiler import ProfilerActivity, profile
    except ImportError as exc:
        return {"bytes": None, "flops": None, "status": TORCH_UNAVAILABLE, "error": str(exc)}
    if cuda and not torch.cuda.is_available():
        return {
            "bytes": None,
            "flops": None,
            "status": TORCH_UNAVAILABLE,
            "error": "CUDA requested for the profiler but torch.cuda.is_available() is false",
        }
    device = torch.device("cuda" if cuda else "cpu")
    dtype = torch.float16 if cuda else torch.float32
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    a = torch.randn(m, k, generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)
    b = torch.randn(k, n, generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)
    activities = [ProfilerActivity.CPU]
    if cuda:
        activities.append(ProfilerActivity.CUDA)
        torch.cuda.synchronize()
    try:
        with profile(activities=activities, profile_memory=True, record_shapes=True, with_flops=True) as prof:
            c = a @ b
            if cuda:
                torch.cuda.synchronize()
            _ = c.reshape(-1)[0].item()
    except Exception as exc:  # profiler builds differ; keep the timed sweep
        return {"bytes": None, "flops": None, "status": TORCH_UNAVAILABLE, "error": str(exc)}
    cuda_bytes = 0
    cpu_bytes = 0
    flops = 0.0
    for event in prof.key_averages():
        cuda_bytes += max(0, int(getattr(event, "self_cuda_memory_usage", 0) or 0))
        cpu_bytes += max(0, int(getattr(event, "self_cpu_memory_usage", 0) or 0))
        flops += float(getattr(event, "flops", 0) or 0)
    nbytes = cuda_bytes if cuda else cpu_bytes
    if nbytes <= 0:
        return {"bytes": None, "flops": flops or None, "status": TORCH_EMPTY, "error": None}
    return {"bytes": float(nbytes), "flops": flops or None, "status": TORCH_MEMORY, "error": None}


def _field(fields: dict[str, str], names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in fields:
            return fields[name]
    lowered = {key.lower(): value for key, value in fields.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip().strip('"').replace(",", "")
    if not text or text.lower() in {"n/a", "nan", "-"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None
