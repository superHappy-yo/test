"""Independent, correctness-gated supplementary GPU experiments."""
from __future__ import annotations

import importlib.metadata
import json
import math
import random
import statistics
import subprocess
import time
from pathlib import Path

from shapefuse.bakeoff import DECODE_PREFILL_ATTN, DECODE_PREFILL_GEMM, _torch_rmsnorm
from shapefuse.executor import PlanExecutor, SCOPES, graph_callable, measure, source_fingerprint
from shapefuse.io_utils import write_csv, write_json
from shapefuse.machines import get_machine
from shapefuse.scic import Site, classify_site


def operator_cases():
    cases = [{"label": s.label, "op": "gemm", "M": s.m, "N": s.n, "K": s.k}
             for s in DECODE_PREFILL_GEMM]
    for width in (1024, 8192):
        for m in (1, 128, 512, 2048):
            cases.append(dict(label=f"gemm_d{width}_m{m}", op="gemm", M=m, N=width, K=width))
    for k, n in ((1024, 3072), (3072, 1024)):
        for m in (1, 128, 512):
            cases.append(dict(label=f"gemm_k{k}_n{n}_m{m}", op="gemm", M=m, N=n, K=k))
    for s in DECODE_PREFILL_ATTN:
        cases.append(dict(label=s.label, op="attention", B=s.b, H=s.h, Hkv=s.h,
                          Sq=s.sq, Sk=s.s, D=s.d, causal=False))
    for seq in (512, 2048):
        cases.append(dict(label=f"causal_s{seq}_d128", op="attention", B=1, H=8, Hkv=8,
                          Sq=seq, Sk=seq, D=128, causal=True))
    for seq in (128, 2048):
        # All supplied KV entries are at or before the final query position.
        cases.append(dict(label=f"gqa_b8_sk{seq}", op="attention", B=8, H=16, Hkv=8,
                          Sq=1, Sk=seq, D=64, causal=False))
    return cases


def report_for(case):
    if case["op"] == "gemm":
        shape = {k: case[k] for k in ("M", "N", "K")}
        kind = "gemm"
    elif case["Sq"] == 1:
        shape = {k: case[k] for k in ("B", "H", "Hkv", "Sk", "Sq", "D")}
        kind = "attention_decode"
    else:
        shape = dict(B=case["B"], H=case["H"], S=case["Sk"], D=case["D"])
        kind = "attention_prefill"
    return classify_site(Site(kind, case["label"], shape), get_machine("A40"))


def make_inputs(case, seed, scale=1.0):
    import torch
    gen = torch.Generator(device="cpu").manual_seed(seed)
    def draw(shape):
        return (torch.randn(shape, generator=gen) * scale).to("cuda", torch.float16)
    if case["op"] == "gemm":
        return (draw((case["M"], case["K"])), draw((case["K"], case["N"])),
                torch.ones(case["N"], device="cuda", dtype=torch.float16))
    return (draw((case["B"], case["H"], case["Sq"], case["D"])),
            draw((case["B"], case["Hkv"], case["Sk"], case["D"])),
            draw((case["B"], case["Hkv"], case["Sk"], case["D"])),
            torch.ones(case["D"], device="cuda", dtype=torch.float16))


def native_gemm(a, b, w):
    return _torch_rmsnorm(a @ b, w)


def attention_functions(case):
    import torch
    import torch.nn.functional as F
    def expanded(k, v):
        groups = case["H"] // case["Hkv"]
        return k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
    def materialized(q, k, v, w):
        k, v = expanded(k, v)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        if case["causal"]:
            mask = torch.ones((q.shape[-2], k.shape[-2]), device=q.device, dtype=torch.bool)
            mask = mask.tril(diagonal=k.shape[-2] - q.shape[-2])
            scores = scores.masked_fill(~mask, float("-inf"))
        return _torch_rmsnorm(torch.softmax(scores, -1) @ v, w)
    def native(q, k, v, w):
        y = F.scaled_dot_product_attention(q, k, v, is_causal=case["causal"],
                                          enable_gqa=case["H"] != case["Hkv"])
        return _torch_rmsnorm(y, w)
    def forced_flash(q, k, v, w):
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return native(q, k, v, w)
    def fa2(q, k, v, w):
        from flash_attn import flash_attn_func
        y = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                           dropout_p=0.0, causal=case["causal"]).transpose(1, 2)
        return _torch_rmsnorm(y, w)
    optimized_norm=compiled(_torch_rmsnorm,"default")
    def fa2_compiled_norm(q,k,v,w):
        from flash_attn import flash_attn_func
        y=flash_attn_func(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
                         dropout_p=0.0,causal=case["causal"]).transpose(1,2)
        return optimized_norm(y,w)
    def reference(q, k, v, w):
        return materialized(q.float(), k.float(), v.float(), w.float()).half()
    return native, reference, {"materialized": materialized, "sdpa_flash": forced_flash,
                              "fa2": fa2,"fa2_compiled_norm":fa2_compiled_norm}


def compiled(fn, mode):
    import torch
    if mode == "default":
        return torch.compile(fn, fullgraph=True, options={"triton.cudagraphs": False})
    return torch.compile(fn, fullgraph=True, mode=mode)


def audit(fn, inputs, reference, *, attention=False):
    import torch
    with torch.inference_mode():
        expected = reference(*inputs)
        actual = fn(*inputs)
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(actual).all())
        tol = 1e-2 if attention else 1e-3
        passed = finite and bool(torch.allclose(actual, expected, rtol=tol, atol=tol))
        return {"passed": passed, "finite": finite, "atol": tol, "rtol": tol,
                "max_abs_error": float((actual.float() - expected.float()).abs().max()),
                "collapsed_rows": int((actual == 0).all(-1).sum()),
                "reference_collapsed_rows": int((expected == 0).all(-1).sum())}


def inspect_kernels(fn, inputs):
    import torch
    try:
        with torch.inference_mode(), torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            fn(*inputs)
            torch.cuda.synchronize()
        events = prof.events()
        kernels = [e.name for e in events if str(e.device_type).endswith("CUDA")]
        backends = sorted({e.name for e in events if "_scaled_dot_product" in e.name})
        return {"status": "ok", "kernels": kernels, "kernel_count": len(kernels),
                "sdpa_backend_ops": backends, "dram_bytes": None}
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc), "dram_bytes": None}


def environment():
    import torch
    packages = {}
    for name in ("torch", "triton", "transformers", "flash-attn", "vllm", "tilelang"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu",
                            "--format=csv,noheader"], capture_output=True, text=True)
    return {"packages": packages, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
            "gpu_snapshot": query.stdout, "visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
            "source_hash": source_fingerprint(Path(__file__), Path(__file__).with_name("executor.py"),
                                              Path(__file__).with_name("bakeoff.py")),
            "thresholds": [0.5, 2.0], "ridge_source": "datasheet", "quote_as_scic_accuracy": False}


def run_operator(args, *, research=False):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Supplementary experiments require CUDA; no CPU result substituted")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "manifest.json", environment())
    for key in ("cache_size_limit", "recompile_limit"):
        if hasattr(torch._dynamo.config, key):
            setattr(torch._dynamo.config, key, 128)
    rows, details = [], []
    cases = [c for c in operator_cases() if (not args.case or c["label"] == args.case)
             and (not research or c["op"] == "attention")]
    if not cases:
        raise ValueError(f"Unknown case: {args.case}")
    for case in cases:
        if research and case["op"] != "attention":
            continue
        print("CASE", case["label"], flush=True)
        inputs = make_inputs(case, args.seed)
        if case["op"] == "gemm":
            native, reference, extra = native_gemm, native_gemm, {}
        else:
            native, reference, extra = attention_functions(case)
        recipes = {"native": lambda: native, "compile_default": lambda: compiled(native, "default"),
                   "compile_max_no_graph": lambda: compiled(native, "max-autotune-no-cudagraphs")}
        if args.graph_mode == "compiler":
            recipes.update({"compile_reduce_overhead": lambda: compiled(native, "reduce-overhead"),
                            "compile_max_autotune": lambda: compiled(native, "max-autotune")})
        elif not research:
            recipes.update({name: (lambda fn=fn: fn) for name, fn in extra.items()})
        if research:
            from shapefuse.research import research_recipes
            recipes.update(research_recipes(case))
        allowed = set(args.methods.split(",")) if args.methods else None
        validated = {}
        candidate_order = list(recipes.items())
        random.Random(args.seed).shuffle(candidate_order)
        for name, factory in candidate_order:
            if allowed and name not in allowed:
                continue
            row = dict(case=case["label"], op=case["op"], method=name, graph_mode=args.graph_mode,
                       seed=args.seed, execution_status="failed", timing_source="executed",
                       copied_from="", selected_candidate="", **report_for(case).__dict__)
            record = {"case": case, "method": name, "validation": [],
                      "candidate_order": [n for n, _ in candidate_order],
                      "order_protocol": "seeded shuffle"}
            try:
                start = time.perf_counter()
                if name == "neptune" and args.graph_mode == "manual":
                    raise NotImplementedError(
                        "This TVM/Triton bridge crashed with SIGSEGV under CUDA Graph capture; "
                        "direct-call results remain separate. See archived process logs.")
                fn = factory()
                for seed, scale in ((args.seed, 1.0), (args.seed + 17, 1.0),
                                    (args.seed + 31, 5.0 if case["op"] == "gemm" else 1.0)):
                    xs = inputs if seed == args.seed else make_inputs(case, seed, scale)
                    checked = audit(fn, xs, reference, attention=case["op"] == "attention")
                    record["validation"].append(dict(seed=seed, scale=scale, **checked))
                    if not checked["passed"]:
                        raise ArithmeticError(f"Full-output validation failed: {checked}")
                    del xs
                row["setup_ms"] = (time.perf_counter() - start) * 1000
                record["inspection"] = inspect_kernels(fn, inputs)
                if args.graph_mode == "manual":
                    fn = graph_callable(fn, inputs)
                validated[name] = fn
                record["measurement"] = measure(fn, inputs, warmup=args.warmup or 50,
                                                 repeats=args.repeats or 200)
                row.update({k: v for k, v in record["measurement"].items() if k != "gpu_samples_ms"})
                row.update(execution_status="ok", error="")
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                record["error"] = row["error"]
                torch.cuda.synchronize()
            rows.append(row)
            details.append(record)
            print(name, row["execution_status"], row.get("gpu_median_ms"), flush=True)
        if args.graph_mode != "compiler" and "native" in validated and "compile_default" in validated:
            # Every policy invokes its own dispatcher, with independent timing.
            for scope in SCOPES:
                executor = PlanExecutor(validated, scope=scope, implementation=environment()["source_hash"],
                                        device_identity=str(torch.cuda.get_device_properties(0)),
                                        attention=case["op"] == "attention")
                prepared = executor.prepare(report_for(case), inputs)
                checked = audit(executor, inputs, reference, attention=case["op"] == "attention")
                if not checked["passed"]:
                    raise ArithmeticError("Dispatcher output failed validation")
                measured = measure(executor, inputs, warmup=args.warmup or 50, repeats=args.repeats or 200)
                row = dict(case=case["label"], op=case["op"], method=f"scic_{scope}",
                           graph_mode=args.graph_mode, seed=args.seed, execution_status="ok",
                           timing_source="executed", copied_from="", selected_candidate=prepared.candidate,
                           calibration_ms=prepared.calibration_ms, setup_ms=0, error="",
                           **report_for(case).__dict__,
                           **{k: v for k, v in measured.items() if k != "gpu_samples_ms"})
                rows.append(row)
                details.append({"case": case, "method": row["method"], "validation": [checked],
                                "selection": prepared.__dict__ | {"fn": None}, "measurement": measured,
                                "cache_hits": executor.hits, "cache_misses": executor.misses})
                print(row["method"], prepared.candidate, row["gpu_median_ms"], flush=True)
        write_csv(out / "latency.csv", rows)
        write_json(out / "details.json", details)
        del inputs, validated
        torch._dynamo.reset()
        torch.cuda.empty_cache()
    write_json(out / "completed.json", {"cases": len(cases), "rows": len(rows),
                                      "ok": sum(r["execution_status"] == "ok" for r in rows)})
    return 0


def main_supplement(args):
    if args.stage in ("operator", "research"):
        return run_operator(args, research=args.stage == "research")
    if args.stage == "system":
        if args.framework == "vllm":
            from shapefuse.vllm_system import run_vllm
            return run_vllm(args)
        from shapefuse.model_system import run_pytorch
        return run_pytorch(args)
    base = Path(args.out)
    for stage in ("operator", "research", "system"):
        import copy
        child = copy.copy(args)
        child.stage, child.out = stage, str(base / stage)
        main_supplement(child)
    return 0
