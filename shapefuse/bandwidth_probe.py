"""Copy-kernel bandwidth estimate. Not Nsight, and not a measured ridge.

Times a device-to-device ``copy_`` and divides bytes moved by seconds.
Bytes moved count a read and a write. The ridge estimate divides the
**datasheet** peak FLOP/s by that bandwidth. Both numbers are labeled
``ESTIMATE_COPY_NOT_NSIGHT``. They must not be copied into
``measured_hbm_bandwidth_bytes_per_s`` or ``measured_ridge_flops_per_byte``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from shapefuse.machines import get_machine

ESTIMATE_STATUS = "ESTIMATE_COPY_NOT_NSIGHT"
BYTES_ACCOUNTING = "device_to_device_copy_read_plus_write"


def estimate_from_timing(
    *,
    seconds: float,
    repeats: int,
    numel: int,
    element_bytes: int,
    datasheet_peak_flops_per_s: float,
) -> dict:
    """Build the estimate payload from an already measured copy time."""
    if seconds <= 0:
        raise ValueError(f"copy time must be positive, got {seconds}")
    if repeats <= 0 or numel <= 0 or element_bytes <= 0:
        raise ValueError("repeats, numel, and element_bytes must be positive")
    if datasheet_peak_flops_per_s <= 0:
        raise ValueError("datasheet peak must be positive")
    bytes_moved = repeats * 2 * numel * element_bytes
    bandwidth = bytes_moved / seconds
    return {
        "status": ESTIMATE_STATUS,
        "not_nsight": True,
        "not_a_measured_ridge": True,
        "do_not_write_measured_ridge_flops_per_byte": True,
        "bytes_accounting": BYTES_ACCOUNTING,
        "repeats": repeats,
        "numel": numel,
        "element_bytes": element_bytes,
        "bytes_moved": bytes_moved,
        "seconds": seconds,
        "estimated_bandwidth_bytes_per_s": bandwidth,
        "estimated_bandwidth_gb_per_s": bandwidth / 1e9,
        "datasheet_peak_flops_per_s": datasheet_peak_flops_per_s,
        "ridge_numerator": "datasheet_peak_flops_per_s",
        "ridge_denominator": "copy_kernel_bandwidth_estimate",
        "estimated_ridge_flops_per_byte": datasheet_peak_flops_per_s / bandwidth,
        "measured_hbm_bandwidth_bytes_per_s": None,
        "measured_ridge_flops_per_byte": None,
        "note": (
            "ESTIMATE from a timed device-to-device copy, counting read plus "
            "write. The ridge numerator is the datasheet FLOP peak, not a "
            "measured peak. This is not Nsight DRAM and must not replace "
            "measured_ridge_flops_per_byte."
        ),
    }


def measure_copy_bandwidth(numel: int, repeats: int, warmup: int = 5) -> tuple[float, int]:
    """Time ``b.copy_(a)`` on CUDA. Returns ``(seconds, element_bytes)``."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing to invent a bandwidth")
    a = torch.empty(numel, device="cuda", dtype=torch.float16)
    b = torch.empty_like(a)
    # Touch both buffers so the timed loop is not first-touch paging.
    a.fill_(1)
    b.zero_()
    torch.cuda.synchronize()
    for _ in range(warmup):
        b.copy_(a)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        b.copy_(a)
    end.record()
    torch.cuda.synchronize()
    seconds = start.elapsed_time(end) / 1e3
    return float(seconds), int(a.element_size())


def write_estimate(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Estimate GDDR bandwidth with a copy kernel. Not Nsight.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--machine", default="A40_FP16")
    parser.add_argument("--numel", type=int, default=256 * 1024 * 1024, help="FP16 elements per buffer")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        machine = get_machine(args.machine)
    except KeyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is not installed. Not writing a bandwidth estimate.", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Not writing a bandwidth estimate.", file=sys.stderr)
        return 2
    try:
        seconds, element_bytes = measure_copy_bandwidth(args.numel, args.repeats)
        payload = estimate_from_timing(
            seconds=seconds,
            repeats=args.repeats,
            numel=args.numel,
            element_bytes=element_bytes,
            datasheet_peak_flops_per_s=machine.peak_flops_per_s,
        )
    except Exception as exc:
        print(f"ERROR: bandwidth probe failed: {exc}", file=sys.stderr)
        return 2
    payload["device_name"] = torch.cuda.get_device_name(0)
    payload["machine"] = machine.name
    payload["datasheet_peak_source"] = machine.source
    write_estimate(args.out, payload)
    print(
        f"wrote {args.out} status={payload['status']} "
        f"estimated_bandwidth_gb_per_s={payload['estimated_bandwidth_gb_per_s']:.3f} "
        f"(not Nsight, not a measured ridge)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
