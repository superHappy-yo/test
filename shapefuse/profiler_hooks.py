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
TORCH_ALLOCATOR_DELTA = "TORCH_ALLOCATOR_DELTA_NOT_DRAM"
TORCH_EMPTY = "TORCH_PROFILER_EMPTY"
TORCH_UNAVAILABLE = "TORCH_PROFILER_UNAVAILABLE"
PROFILER_OFF = "PROFILER_OFF"
TBD_NSIGHT = "TBD_NSIGHT"

# Kineto chrome-trace device type for CUDA. CPU is 0.
_CHROME_CUDA_DEVICE = 1

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

    ``bytes`` is allocator memory, not DRAM traffic. ``status`` is never an
    Nsight status. PyTorch 2.5 records that allocator delta on
    ``self_device_memory_usage``. Reading only the deprecated
    ``self_cuda_memory_usage`` name yields zeros on ``key_averages()`` and
    was reported as ``TORCH_PROFILER_EMPTY`` on the A40 package.

    If profiler events are still empty, a positive caching-allocator delta
    is returned with status ``TORCH_ALLOCATOR_DELTA_NOT_DRAM``. That delta
    is also not DRAM and not achieved AI.
    """
    try:
        import torch
        from torch.profiler import ProfilerActivity, profile
    except ImportError as exc:
        return _proxy_result(None, None, TORCH_UNAVAILABLE, str(exc), "")
    if cuda and not torch.cuda.is_available():
        return _proxy_result(
            None,
            None,
            TORCH_UNAVAILABLE,
            "CUDA requested for the profiler but torch.cuda.is_available() is false",
            "",
        )
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
    alloc_before = None
    alloc_after = None
    try:
        # with_stack is required by torch.profiler.export_memory_timeline.
        # record_shapes and profile_memory are what attach allocator events.
        # Older profilers reject with_stack; retry without it.
        try:
            prof_cm = profile(
                activities=activities,
                profile_memory=True,
                record_shapes=True,
                with_stack=True,
                with_flops=True,
            )
        except TypeError:
            prof_cm = profile(
                activities=activities,
                profile_memory=True,
                record_shapes=True,
                with_flops=True,
            )
        with prof_cm as prof:
            if cuda:
                torch.cuda.synchronize()
                alloc_before = int(torch.cuda.memory_allocated())
            c = a @ b
            if cuda:
                torch.cuda.synchronize()
                alloc_after = int(torch.cuda.memory_allocated())
            _ = c.reshape(-1)[0].item()
            if cuda:
                alloc_after = max(alloc_after, int(torch.cuda.memory_allocated()))
    except Exception as exc:  # profiler builds differ; keep the timed sweep
        return _proxy_result(None, None, TORCH_UNAVAILABLE, str(exc), "")
    grouped, raw = _grouped_and_raw_events(prof)
    flops = _flops_from_events(grouped or raw)
    event_bytes, _event_source = bytes_from_profiler_events(grouped, cuda=cuda)
    if event_bytes is None or event_bytes <= 0:
        event_bytes, _event_source = bytes_from_profiler_events(raw, cuda=cuda)
    # Chrome export is only a fallback. Skip it when the event table already
    # has a positive allocator byte count.
    trace = None if event_bytes is not None and event_bytes > 0 else _chrome_trace_text(prof)
    chosen = collect_proxy_bytes(
        events=grouped,
        raw_events=raw,
        chrome_trace=trace,
        allocator_delta=_allocator_delta(alloc_before, alloc_after) if cuda else None,
        cuda=cuda,
    )
    if chosen["bytes"] is None:
        return _proxy_result(None, flops, TORCH_EMPTY, chosen.get("error"), "")
    return _proxy_result(chosen["bytes"], flops, chosen["status"], None, chosen["source"])


def collect_proxy_bytes(
    *,
    events,
    chrome_trace: str | None,
    allocator_delta: int | None,
    cuda: bool,
    raw_events=None,
) -> dict:
    """Pick a byte count without promoting it to DRAM or achieved AI.

    Order: profiler event fields (averaged, then raw), then ``[memory]``
    events in a chrome trace, then the caching-allocator delta. The last
    one uses a different status so a later reader can see that the profiler
    table was empty.
    """
    nbytes, source = bytes_from_profiler_events(events, cuda=cuda)
    if (nbytes is None or nbytes <= 0) and raw_events:
        nbytes, source = bytes_from_profiler_events(raw_events, cuda=cuda)
        if nbytes is not None and nbytes > 0 and "." in source:
            source = "events." + source.split(".", 1)[1]
    if nbytes is not None and nbytes > 0:
        return {"bytes": float(nbytes), "status": TORCH_MEMORY, "source": source, "error": None}
    if chrome_trace:
        traced = chrome_trace_memory_net(chrome_trace, cuda=cuda)
        if traced is not None and traced > 0:
            return {
                "bytes": float(traced),
                "status": TORCH_MEMORY,
                "source": "chrome_trace.memory",
                "error": None,
            }
    if allocator_delta is not None and allocator_delta > 0:
        return {
            "bytes": float(allocator_delta),
            "status": TORCH_ALLOCATOR_DELTA,
            "source": "caching_allocator_delta",
            "error": None,
        }
    return {"bytes": None, "status": TORCH_EMPTY, "source": "", "error": None}


def bytes_from_profiler_events(events, cuda: bool) -> tuple[int | None, str]:
    """Read allocator bytes from profiler events.

    PyTorch 2.5 ``key_averages()`` exposes ``self_device_memory_usage`` and
    does not define ``self_cuda_memory_usage`` (that name remains only as a
    deprecated alias on the un-averaged event). Missing the new field is what
    made the A40 run record ``TORCH_PROFILER_EMPTY``.
    """
    self_names = (
        ("self_device_memory_usage", "self_cuda_memory_usage", "self_privateuse1_memory_usage")
        if cuda
        else ("self_cpu_memory_usage",)
    )
    total_names = (
        ("device_memory_usage", "cuda_memory_usage", "privateuse1_memory_usage")
        if cuda
        else ("cpu_memory_usage",)
    )
    self_sum = 0
    saw_self = False
    best_total = 0
    saw_total = False
    for event in events:
        self_value = _first_int(event, self_names)
        if self_value is not None:
            saw_self = True
            self_sum += self_value
        total_value = _first_int(event, total_names)
        if total_value is not None:
            saw_total = True
            if total_value > best_total:
                best_total = total_value
    if saw_self and self_sum > 0:
        return self_sum, "key_averages.self_device_memory_usage" if cuda else "key_averages.self_cpu_memory_usage"
    if saw_total and best_total > 0:
        return best_total, "key_averages.device_memory_usage" if cuda else "key_averages.cpu_memory_usage"
    return None, ""


def chrome_trace_memory_net(text: str, cuda: bool) -> int | None:
    """Net ``[memory]`` bytes from a profiler chrome trace.

    Frees are negative ``Bytes``. The result is allocator traffic in the
    trace, not DRAM. Returns None when the trace has no matching events.
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        events = payload.get("traceEvents", [])
    elif isinstance(payload, list):
        events = payload
    else:
        return None
    if not isinstance(events, list):
        return None
    net = 0
    saw = False
    for event in events:
        if not isinstance(event, dict) or event.get("name") != "[memory]":
            continue
        args = event.get("args") or {}
        if not isinstance(args, dict):
            continue
        if not _chrome_device_matches(args.get("Device Type", args.get("device_type")), cuda):
            continue
        nbytes = _float_or_none(args.get("Bytes", args.get("bytes")))
        if nbytes is None:
            continue
        net += int(nbytes)
        saw = True
    if not saw:
        return None
    return net


def _proxy_result(nbytes, flops, status: str, error, source: str) -> dict:
    return {
        "bytes": None if nbytes is None else float(nbytes),
        "flops": flops or None,
        "status": status,
        "error": error,
        "source": source,
    }


def _grouped_and_raw_events(prof) -> tuple[list, list]:
    try:
        grouped = list(prof.key_averages())
    except Exception:
        grouped = []
    try:
        raw = list(prof.events())
    except Exception:
        raw = []
    return grouped, raw


def _flops_from_events(events) -> float | None:
    total = 0.0
    for event in events:
        total += float(getattr(event, "flops", 0) or 0)
    return total or None


def _chrome_trace_text(prof) -> str | None:
    import tempfile

    try:
        with tempfile.NamedTemporaryFile(suffix=".json") as handle:
            prof.export_chrome_trace(handle.name)
            return Path(handle.name).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def _allocator_delta(before: int | None, after: int | None) -> int | None:
    if before is None or after is None:
        return None
    return int(after) - int(before)


def _first_int(event, names: tuple[str, ...]) -> int | None:
    missing = object()
    for name in names:
        value = getattr(event, name, missing)
        if value is missing:
            continue
        parsed = _coerce_int(value)
        if parsed is not None:
            return parsed
    return None


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, dict):
        total = 0
        saw = False
        for item in value.values():
            parsed = _coerce_int(item)
            if parsed is None:
                continue
            total += parsed
            saw = True
        return total if saw else None
    return None


def _chrome_device_matches(device_type: object, cuda: bool) -> bool:
    if device_type is None:
        return True
    if isinstance(device_type, str):
        text = device_type.strip().lower()
        if text in {"cuda", "gpu", "1"}:
            return cuda
        if text in {"cpu", "0"}:
            return not cuda
        return True
    try:
        code = int(device_type)
    except (TypeError, ValueError):
        return True
    if code == _CHROME_CUDA_DEVICE:
        return cuda
    if code == 0:
        return not cuda
    return False


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
