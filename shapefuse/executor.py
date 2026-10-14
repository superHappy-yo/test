"""Executable SCIC bindings. Latency is measured by the caller, never copied.

The binding vocabulary remains compiled chain / native chain. These bindings
do not claim that an entire GEMM reduction was fused into a single kernel.
"""
from __future__ import annotations

import hashlib
import statistics
import time
from dataclasses import dataclass

from shapefuse.plans import select_plan

SCOPES = ("off", "ridge-only", "ridge-compute", "all-gemm")


def tensor_signature(tensor):
    return (tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype), str(tensor.device))


def exact_key(inputs, implementation, device_identity):
    """Exact layouts supplement, rather than replace, the analytical bucket."""
    return (implementation, device_identity, tuple(tensor_signature(x) for x in inputs))


def calibration_applies(regime, scope):
    if scope not in SCOPES:
        raise ValueError(f"Unknown calibration scope: {scope}")
    return scope == "all-gemm" or (scope == "ridge-compute" and regime != "memory-bound") or (
        scope == "ridge-only" and regime == "ridge-region"
    )


def measure(fn, inputs, *, warmup=50, repeats=200):
    """Individual GPU events and a separate synchronized host batch.

The host batch includes function dispatch and launch costs. Its single final
barrier is amortized over repeats; it is not per-call synchronized latency.
"""
    import torch

    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be positive; warmup must be nonnegative")
    with torch.inference_mode():
        for _ in range(warmup):
            fn(*inputs)
        torch.cuda.synchronize()
        events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                  for _ in range(repeats)]
        for start, end in events:
            start.record()
            fn(*inputs)
            end.record()
        torch.cuda.synchronize()
        samples = [start.elapsed_time(end) for start, end in events]
        start_host = time.perf_counter()
        for _ in range(repeats):
            fn(*inputs)
        torch.cuda.synchronize()
        host = (time.perf_counter() - start_host) * 1000 / repeats
    return {"gpu_median_ms": statistics.median(samples), "gpu_samples_ms": samples,
            "host_batch_ms_per_call": host, "repeats": repeats, "warmup": warmup}


def graph_callable(fn, inputs):
    import torch

    # Use a side stream during warmup, and retain input/output storage.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.inference_mode():
        for _ in range(3):
            fn(*inputs)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        output = fn(*inputs)

    def replay(*actual):
        # Benchmark uses the same static storage. Do not silently replay stale
        # inputs when this callable is used elsewhere.
        if len(actual) != len(inputs) or any(a.data_ptr() != b.data_ptr() for a, b in zip(actual, inputs)):
            raise ValueError("CUDA Graph runner requires its captured input storage")
        graph.replay()
        return output

    return replay


@dataclass
class PreparedPath:
    candidate: str
    fn: object
    calibration_ms: float
    probe: dict
    analytical_candidate: str
    regime: str
    bucket: str


class PlanExecutor:
    """Cache a real callable per exact signature and invoke it on every call."""

    def __init__(self, candidates, *, scope="ridge-compute", implementation="", device_identity="",
                 attention=False):
        if scope not in SCOPES:
            raise ValueError(scope)
        self.candidates = candidates
        self.scope = scope
        self.implementation = implementation
        self.device_identity = device_identity
        self.attention = attention
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def prepare(self, report, inputs, *, warmup=3, repeats=10):
        key = exact_key(inputs, self.implementation, self.device_identity)
        if key in self.cache:
            return self.cache[key]
        decision = select_plan(report)
        primary = "compile_default" if self.attention or decision.primary.mode == "MemMode" else "native"
        available = {k: v for k, v in self.candidates.items() if k in ("native", "compile_default")}
        if not available:
            raise RuntimeError("No validated SCIC candidate")
        chosen = primary if primary in available else next(iter(available))
        probe = {}
        start = time.perf_counter()
        if not self.attention and calibration_applies(report.regime, self.scope) and len(available) == 2:
            # Alternate the probe order across shape dimensions; regular
            # timing is a later, independent sample.
            names = sorted(available, reverse=bool(sum(inputs[0].shape) % 2))
            for name in names:
                probe[name] = measure(available[name], inputs, warmup=warmup, repeats=repeats)["gpu_median_ms"]
            if probe[names[0]] != probe[names[1]]:
                chosen = min(probe, key=probe.get)
        elapsed = (time.perf_counter() - start) * 1000 if probe else 0.0
        prepared = PreparedPath(chosen, available[chosen], elapsed, probe, primary, report.regime,
                                decision.cache_key)
        self.cache[key] = prepared
        return prepared

    def __call__(self, *inputs):
        key = exact_key(inputs, self.implementation, self.device_identity)
        try:
            prepared = self.cache[key]
        except KeyError as exc:
            self.misses += 1
            raise RuntimeError("SCIC signature not prepared; calibrate before timing/capture") from exc
        self.hits += 1
        return prepared.fn(*inputs)


def source_fingerprint(*paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
