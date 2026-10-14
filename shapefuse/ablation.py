"""Controlled diagnostics of the HF adapter, not new kernel schedules.

The prebound variants are specializations of the explicitly fixed prefill and
decode workloads. They are deliberately separate from the general exact-key
executor: their caller checks the phase shape, and the outer compiled model
uses static guards. Preparation and numerical validation precede freezing.
"""
from __future__ import annotations

import copy
import random
import time
from pathlib import Path

from shapefuse import model_system as ms
from shapefuse.executor import exact_key, measure, source_fingerprint
from shapefuse.io_utils import write_csv, write_json
from shapefuse.supplement import environment

POLICIES = ("original", "fixed_compile", "fixed_compile_single",
            "fixed_compile_bound", "inline_strict", "scic", "scic_bound")
OUTER_OPTIONS = {"triton.cudagraphs": False, "emulate_precision_casts": True}


class LegacyLookupSite(ms.ProjectionSite):
    """Retain the archived two-lookup hot path after promoting the fix."""
    def invoke(self, *inputs):
        key = exact_key(inputs, self.executor.implementation, self.executor.device_identity)
        if key not in self.executor.cache:
            return super().invoke(*inputs)
        return self.executor(*inputs)


class SingleLookupSite(ms.ProjectionSite):
    def invoke(self, *inputs):
        key = exact_key(inputs, self.executor.implementation, self.executor.device_identity)
        entry = self.executor.cache.get(key)
        if entry is None:
            return super().invoke(*inputs)
        self.executor.hits += 1
        return entry.fn(*inputs)


class BoundEndpoint:
    """An internal endpoint for one prepared signature in one model phase."""
    def __init__(self, policy, forced_candidate=None):
        self.site = ms.ProjectionSite(policy)
        self.entry = None
        self.signature = None
        self.frozen = False
        self.calls = 0
        self.forced_candidate = forced_candidate
        self.site_id = len(ms._SITES)
        ms._SITES[self.site_id] = self

    def invoke(self, *inputs):
        if self.frozen:
            self.calls += 1
            return self.entry.fn(*inputs)
        key = exact_key(inputs, self.site.executor.implementation, self.site.executor.device_identity)
        if self.signature is not None and key != self.signature:
            raise ValueError("A bound phase must have exactly one tensor signature")
        output = self.site.invoke(*inputs)
        self.signature = key
        self.entry = self.site.executor.cache[key]
        # Reuse the legacy SCIC decision to isolate invocation overhead. Never
        # infer the decision from this variant's regular timing samples.
        if self.forced_candidate is not None:
            candidate = self.forced_candidate
            self.entry.candidate = candidate
            self.entry.fn = self.site.executor.candidates[candidate]
            self.entry.fn(*inputs)
            output = self.entry.fn(*inputs)
            import torch
            reference = self.site.native(*inputs)
            if not all(torch.isfinite(a).all() and torch.allclose(a, b, rtol=1e-3, atol=1e-3)
                       for a, b in zip(output, reference)):
                raise ArithmeticError("Forced prebound candidate failed component validation")
        return output

    def freeze(self):
        if self.entry is None:
            raise RuntimeError("Prepare and validate the endpoint before freezing")
        self.frozen = True


def adapt(model, policy, batch, prompt_length, decisions=None):
    import torch
    ms.register_operator()

    class Layer(torch.nn.Module):
        def __init__(self, layer, index):
            super().__init__()
            self.self_attn, self.mlp = layer.self_attn, layer.mlp
            self.input_layernorm = layer.input_layernorm
            self.post_attention_layernorm = layer.post_attention_layernorm
            self.projection = self.self_attn.o_proj
            self.self_attn.o_proj = torch.nn.Identity()
            if policy.endswith("_bound"):
                # Each phase has its own endpoint; no shape-to-path map is
                # consulted inside the opaque operation after freezing.
                self.endpoints = [BoundEndpoint("fixed_compile",
                    decisions[(index, phase)] if decisions is not None else None)
                    for phase in ("prefill", "decode")]
            elif policy != "inline_strict":
                cls = SingleLookupSite if policy.endswith("_single") else LegacyLookupSite
                self.site = cls("scic" if policy == "scic" else "fixed_compile")

        def forward(self, hidden_states, attention_mask=None, position_ids=None,
                    past_key_value=None, output_attentions=False, use_cache=False,
                    cache_position=None, position_embeddings=None, **kwargs):
            residual = hidden_states
            if residual.shape[0] != batch or residual.shape[1] not in (1, prompt_length):
                raise ValueError("Static ablation model received an unprepared phase shape")
            hidden_states, weights = self.self_attn(
                hidden_states=self.input_layernorm(hidden_states), attention_mask=attention_mask,
                position_ids=position_ids, past_key_value=past_key_value,
                output_attentions=output_attentions, use_cache=use_cache,
                cache_position=cache_position, position_embeddings=position_embeddings, **kwargs)
            shape = residual.shape
            inputs = (hidden_states.reshape(-1, hidden_states.shape[-1]), self.projection.weight,
                      residual.reshape(-1, residual.shape[-1]), self.post_attention_layernorm.weight)
            if policy == "inline_strict":
                normalized, summed = ms.project_add_norm_strict(*inputs)
            else:
                site_id = (self.endpoints[1 if shape[1] == 1 else 0].site_id
                           if policy.endswith("_bound") else self.site.site_id)
                normalized, summed = torch.ops.shapefuse.project_add_norm(*inputs, site_id)
            hidden_states = self.mlp(normalized.view(shape)) + summed.view(shape)
            return (hidden_states, weights) if output_attentions else (hidden_states,)

    model.model.layers = torch.nn.ModuleList([Layer(layer, i) for i, layer in enumerate(model.model.layers)])
    return model


def output_audit(actual, reference, atol=1e-2, rtol=1e-2):
    import torch
    if len(actual) != len(reference):
        raise ArithmeticError("Output history length mismatch")
    errors = [float((a-b).abs().max()) for a, b in zip(actual, reference)]
    if not all(bool(torch.isfinite(a).all()) and torch.allclose(a, b, atol=atol, rtol=rtol)
               for a, b in zip(actual, reference)):
        raise ArithmeticError(f"Full output validation failed: max_abs={max(errors)}")
    return max(errors)


def profiler_summary(fn):
    import torch
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profile:
        fn()
        torch.cuda.synchronize()
    kernels = {}
    cpu = []
    for event in profile.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            kernels[event.name] = kernels.get(event.name, 0) + 1
    for event in profile.key_averages():
        cpu.append(dict(name=event.key, count=event.count, self_cpu_us=event.self_cpu_time_total))
    return dict(cuda_events=kernels, cuda_event_count=sum(kernels.values()),
                cpu=sorted(cpu, key=lambda x: x["self_cpu_us"], reverse=True)[:30],
                limitation="Profiler adds overhead; these samples do not enter latency summaries or measure DRAM bytes")


def run_system(args):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM, StaticCache
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    batch, length = map(int, args.case.replace("b", "").split("_s"))
    if (batch, length) not in ((1, 128), (1, 2048), (8, 128), (8, 2048)):
        raise ValueError(args.case)
    torch.manual_seed(args.seed)
    config = Qwen3Config(**ms.MODEL_CONFIG)
    config._attn_implementation = "sdpa"
    with torch.inference_mode():
        base = Qwen3ForCausalLM(config).eval().half().cuda()
    gen = torch.Generator().manual_seed(args.seed + batch + length)
    tokens = torch.randint(3, config.vocab_size, (batch, length+64), generator=gen).cuda()
    mask = torch.ones((batch, length+64), device="cuda", dtype=torch.long)
    positions = [torch.arange(length, device="cuda")]
    positions += [torch.tensor([length+i], device="cuda") for i in range(64)]
    policies = args.methods.split(",") if args.methods else list(POLICIES)
    if any(p not in POLICIES for p in policies):
        raise ValueError(policies)
    random.Random(args.seed).shuffle(policies)
    # Binding comparison shares the legacy SCIC decisions. Keep all other
    # comparisons in seeded order, and log this explicit dependency.
    if "scic_bound" in policies:
        if "scic" not in policies:
            raise ValueError("scic_bound requires measured scic in the same process")
        policies.remove("scic_bound")
        policies.insert(policies.index("scic")+1, "scic_bound")
    write_json(out / "manifest.json", environment() | dict(
        model=ms.MODEL_CONFIG, parameters=sum(p.numel() for p in base.parameters()),
        seed=args.seed, case=args.case, policies=policies, pretrained=False,
        outer_options=OUTER_OPTIONS, dynamic=False, positions_preallocated=True,
        timing="Independent CUDA events and synchronized host batches; setup/profiling excluded",
        source_sha256=source_fingerprint(Path(__file__), Path(ms.__file__)),
        distinction_from_previous="All arms emulate precision casts; decode positions are preallocated; compare only within this new protocol"))
    model = None
    cache = None

    def prefill():
        return model(tokens[:, :length], attention_mask=mask, past_key_values=cache,
                     cache_position=positions[0], use_cache=True, logits_to_keep=1).logits

    def decode(collect=False):
        values = []
        for i in range(64):
            y = model(tokens[:, length+i:length+i+1], attention_mask=mask,
                      past_key_values=cache, cache_position=positions[i+1],
                      use_cache=True, logits_to_keep=1).logits
            if collect:
                values.append(y.detach().clone())
        return values

    def new_cache():
        return StaticCache(config=config, max_batch_size=batch, max_cache_len=length+64,
                           device="cuda", dtype=torch.float16)

    rows, details, decisions = [], [], None
    with torch.inference_mode():
        model, cache = base, new_cache()
        references = [prefill().detach().clone(), *decode(True)]
        for policy in policies:
            print("ABLATION", args.case, policy, flush=True)
            record = dict(policy=policy, batch=batch, prompt_length=length, seed=args.seed)
            start = time.perf_counter()
            try:
                model = copy.deepcopy(base)
                if policy != "original":
                    model = adapt(model, policy, batch, length,
                                  decisions if policy == "scic_bound" else None)
                cache = new_cache()
                eager = [prefill().detach().clone(), *decode(True)]
                record["eager_max_abs_logit_error"] = output_audit(eager, references)
                layers = model.model.layers
                if policy.endswith("_bound"):
                    for layer in layers:
                        for endpoint in layer.endpoints:
                            endpoint.freeze()
                if policy == "scic":
                    decisions = {}
                    for i, layer in enumerate(layers):
                        for key, entry in layer.site.executor.cache.items():
                            rows_count = key[2][0][0][0]
                            phase = "decode" if rows_count == batch else "prefill"
                            decisions[i, phase] = entry.candidate
                    if len(decisions) != 12:
                        raise RuntimeError(f"Unexpected SCIC signatures: {decisions}")
                selection = []
                for i, layer in enumerate(layers):
                    if hasattr(layer, "endpoints"):
                        for phase, endpoint in zip(("prefill", "decode"), layer.endpoints):
                            selection.append(dict(layer=i, phase=phase, candidate=endpoint.entry.candidate,
                                                  prepared=endpoint.site.prepared))
                    elif hasattr(layer, "site"):
                        selection += [dict(layer=i, **p) for p in layer.site.prepared]
                record["selection"] = selection
                model = torch.compile(model, dynamic=False, options=OUTER_OPTIONS)
                cache.reset()
                actual = [prefill().detach().clone(), *decode(True)]
                record["compiled_max_abs_logit_error"] = output_audit(actual, references)
                torch.cuda.synchronize()
                record["setup_ms"] = (time.perf_counter()-start)*1000
                pre = measure(prefill, (), warmup=args.warmup, repeats=args.repeats)
                prefill()
                dec = measure(decode, (), warmup=2, repeats=args.repeats)
                for phase, timing in (("prefill", pre), ("decode64", dec)):
                    rows.append(dict(policy=policy, batch=batch, prompt_length=length,
                                     seed=args.seed, phase=phase, execution_status="ok",
                                     **{k: v for k, v in timing.items() if k != "gpu_samples_ms"}))
                    record[phase] = timing
                if args.profile:
                    prefill()
                    record["profile_prefill"] = profiler_summary(prefill)
                    prefill()
                    record["profile_decode64"] = profiler_summary(decode)
                record["execution_status"] = "ok"
                print("PASS", policy, round(record["setup_ms"], 1), flush=True)
            except Exception as exc:
                import traceback
                record.update(execution_status="failed", error=f"{type(exc).__name__}: {exc}",
                              traceback=traceback.format_exc())
                rows.append(dict(policy=policy, batch=batch, prompt_length=length, seed=args.seed,
                                 phase="all", execution_status="failed", error=record["error"]))
                print(record["traceback"], flush=True)
            details.append(record)
            write_csv(out / "latency.csv", rows)
            write_json(out / "details.json", details)
            model = None
            ms._SITES.clear()
            torch._dynamo.reset()
            torch.cuda.empty_cache()
    write_json(out / "completed.json", dict(rows=len(rows), policies=len(policies),
        failed=sum(d["execution_status"] != "ok" for d in details)))
    return int(any(d["execution_status"] != "ok" for d in details))


def run_projection(args):
    import torch
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    ms.register_operator()
    order = [1, 8, 128, 1024, 2048, 16384]
    random.Random(args.seed).shuffle(order)
    write_json(out / "manifest.json", environment() | dict(seed=args.seed, rows=order,
        width=1024, weight_std=0.02, norm_weight="ones, as in the initialized Qwen3 model",
        output_semantics="HF FP16 product/residual; FP32 RMSNorm, cast before affine",
        timing="Same compiled callable across invocation variants; compiler/probe/validation outside timing",
        source_sha256=source_fingerprint(Path(__file__), Path(ms.__file__))))
    records, rows = [], []
    for count in order:
        x = torch.randn(count, 1024, device="cuda", dtype=torch.float16)
        w = torch.randn(1024, 1024, device="cuda", dtype=torch.float16) * 0.02
        residual = torch.randn_like(x)
        norm_w = torch.ones(1024, device="cuda", dtype=torch.float16)
        inputs = x, w, residual, norm_w
        site = LegacyLookupSite("fixed_compile")
        single = SingleLookupSite("fixed_compile")
        with torch.inference_mode():
            site.invoke(*inputs)
            key = exact_key(inputs, site.executor.implementation, site.executor.device_identity)
            entry = site.executor.cache[key]
            single.executor.cache[key] = entry
            endpoint = BoundEndpoint("fixed_compile")
            endpoint.entry, endpoint.signature = entry, key
            endpoint.freeze()
            inline = torch.compile(ms.project_add_norm_strict, fullgraph=True, dynamic=False,
                                   options=OUTER_OPTIONS)
            functions = dict(native=ms.project_add_norm, compiled_direct=entry.fn,
                             legacy_double=site.invoke, single_lookup=single.invoke,
                             prebound=endpoint.invoke,
                             opaque_prebound=lambda *a: torch.ops.shapefuse.project_add_norm(*a, endpoint.site_id),
                             inline_strict=inline)
            names = list(functions)
            random.Random(args.seed+count).shuffle(names)
            reference = ms.project_add_norm(*inputs)
            for name in names:
                fn = functions[name]
                print("PROJECTION", count, name, flush=True)
                record = dict(case=f"m{count}", rows=count, method=name, seed=args.seed)
                try:
                    record["max_abs_error"] = output_audit(fn(*inputs), reference, atol=1e-3, rtol=1e-3)
                    timing = measure(fn, inputs, warmup=args.warmup, repeats=args.repeats)
                    record.update(execution_status="ok", measurement=timing)
                    rows.append(record | {k: v for k,v in timing.items() if k != "gpu_samples_ms" and k != "measurement"})
                    if args.profile:
                        record["profile"] = profiler_summary(lambda: fn(*inputs))
                except Exception as exc:
                    record.update(execution_status="failed", error=f"{type(exc).__name__}: {exc}")
                    rows.append(record)
                records.append(record)
                write_csv(out / "latency.csv", [{k:v for k,v in r.items() if k not in ("measurement", "profile")} for r in rows])
                write_json(out / "details.json", records)
        ms._SITES.clear()
        torch._dynamo.reset()
    write_json(out / "completed.json", dict(rows=len(rows), failed=sum(r["execution_status"] != "ok" for r in rows)))
    return int(any(r["execution_status"] != "ok" for r in rows))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("system", "projection"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--case", default="b1_s128")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    return run_system(args) if args.stage == "system" else run_projection(args)


if __name__ == "__main__":
    raise SystemExit(main())
