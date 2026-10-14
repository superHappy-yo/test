"""Small complete Qwen3 inference, fixed random weights, no checkpoint download."""
from __future__ import annotations

import copy
import statistics
import time
from pathlib import Path

from shapefuse.executor import PlanExecutor, exact_key, measure, source_fingerprint, PreparedPath
from shapefuse.io_utils import write_csv, write_json
from shapefuse.machines import get_machine
from shapefuse.scic import Site, classify_site
from shapefuse.supplement import compiled, environment

MODEL_CONFIG = dict(model_type="qwen3", architectures=["Qwen3ForCausalLM"], hidden_size=1024,
                    intermediate_size=3072, num_hidden_layers=6, num_attention_heads=16,
                    num_key_value_heads=8, head_dim=64, vocab_size=8192,
                    max_position_embeddings=8192, rms_norm_eps=1e-6, tie_word_embeddings=True,
                    attention_bias=False, attention_dropout=0.0, use_cache=True,
                    bos_token_id=1, eos_token_id=2)
POLICIES = ("original", "fixed_native", "fixed_compile", "scic")
_SITES = {}
_REGISTERED = False


def project_add_norm(x, weight, residual, norm_weight):
    import torch
    # HF Qwen3 rounds the residual sum back to FP16 before RMSNorm.
    summed = torch.nn.functional.linear(x, weight) + residual
    values = summed.float()
    normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
    # Match HF's cast-before-affine semantics, distinct from the legacy proxy.
    return (normalized.to(summed.dtype) * norm_weight, summed)


def project_add_norm_strict(x, weight, residual, norm_weight):
    import torch
    # A boundary prevents addmm rewriting from accumulating the residual
    # before the FP16 matrix product has been rounded.
    summed = torch.ops.shapefuse.linear_half(x, weight) + residual
    values = summed.float()
    normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
    return normalized.to(summed.dtype) * norm_weight, summed


def register_operator():
    global _REGISTERED
    if _REGISTERED:
        return
    import torch
    @torch.library.custom_op("shapefuse::linear_half", mutates_args=(),
                            schema="(Tensor x, Tensor weight) -> Tensor")
    def linear_half(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, weight)
    @linear_half.register_fake
    def linear_fake(x, weight):
        return x.new_empty((*x.shape[:-1], weight.shape[0]))
    @torch.library.custom_op("shapefuse::project_add_norm", mutates_args=(),
                            schema="(Tensor x, Tensor weight, Tensor residual, Tensor norm_weight, int site_id) -> (Tensor, Tensor)")
    def op(x: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor,
           norm_weight: torch.Tensor, site_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        return _SITES[site_id].invoke(x, weight, residual, norm_weight)
    @op.register_fake
    def fake(x, weight, residual, norm_weight, site_id):
        return residual.new_empty(residual.shape), residual.new_empty(residual.shape)
    _REGISTERED = True


class ProjectionSite:
    def __init__(self, policy, native=project_add_norm, compile_fn=project_add_norm_strict, identity="hf"):
        import torch
        self.policy = policy
        self.native = native
        # Inductor normally elides intermediate low-precision casts. Preserve
        # the model's explicit FP16 residual rounding instead of relaxing the
        # numerical gate to accommodate that optimization.
        self.compile_fn = None if policy == "fixed_native" else torch.compile(
            compile_fn, fullgraph=True,
            options={"triton.cudagraphs": False, "emulate_precision_casts": True})
        self.executor = PlanExecutor(
            {"native": native, "compile_default": self.compile_fn} if self.compile_fn else {"native": native},
            implementation=f"{identity}|{torch.__version__}|{source_fingerprint(Path(__file__))}",
            device_identity=str(torch.cuda.get_device_properties(0)))
        self.prepared = []
        self.site_id = len(_SITES)
        _SITES[self.site_id] = self

    def invoke(self, *inputs):
        import torch
        key = exact_key(inputs, self.executor.implementation, self.executor.device_identity)
        entry = self.executor.cache.get(key)
        if entry is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Prepare SCIC signature before CUDA Graph capture")
            x, w, _, _ = inputs
            report = classify_site(Site("gemm", f"projection_{self.site_id}",
                                       dict(M=x.numel() // x.shape[-1], K=x.shape[-1], N=w.shape[0])),
                                   get_machine("A40"))
            start = time.perf_counter()
            if self.compile_fn is not None:
                self.compile_fn(*inputs)
                torch.cuda.synchronize()
            if self.policy == "scic":
                entry = self.executor.prepare(report, inputs)
            else:
                candidate = "native" if self.policy == "fixed_native" else "compile_default"
                fn = self.native if candidate == "native" else self.compile_fn
                fn(*inputs)  # Compile outside steady-state timing.
                torch.cuda.synchronize()
                entry = PreparedPath(candidate, fn, 0, {}, candidate, report.regime, "")
                self.executor.cache[key] = entry
            native = self.native(*inputs)
            actual = entry.fn(*inputs)
            passed = all(torch.allclose(a, b, rtol=1e-3, atol=1e-3) and torch.isfinite(a).all()
                         for a, b in zip(actual, native))
            if not passed:
                del self.executor.cache[key]
                errors = [float((a-b).abs().max()) for a,b in zip(actual,native)]
                raise ArithmeticError(f"Projection/residual/norm numerical mismatch: {errors}")
            self.prepared.append(dict(shape=list(x.shape), selected=entry.candidate,
                                      regime=entry.regime, calibration_ms=entry.calibration_ms,
                                      probe=entry.probe, prepare_ms=(time.perf_counter()-start)*1000,
                                      validated=True))
        # The signature has already been checked here. Calling the generic
        # dispatcher would reconstruct the same signature a second time.
        self.executor.hits += 1
        return entry.fn(*inputs)


def adapt_hf(model, policy):
    import torch
    register_operator()
    class AdaptedLayer(torch.nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.self_attn = layer.self_attn
            self.mlp = layer.mlp
            self.input_layernorm = layer.input_layernorm
            self.post_attention_layernorm = layer.post_attention_layernorm
            self.projection = self.self_attn.o_proj
            self.self_attn.o_proj = torch.nn.Identity()
            self.site = ProjectionSite(policy)
        def forward(self, hidden_states, attention_mask=None, position_ids=None,
                    past_key_value=None, output_attentions=False, use_cache=False,
                    cache_position=None, position_embeddings=None, **kwargs):
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states, weights = self.self_attn(
                hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                past_key_value=past_key_value, output_attentions=output_attentions,
                use_cache=use_cache, cache_position=cache_position,
                position_embeddings=position_embeddings, **kwargs)
            # Flatten only the producer's rows. Restore the model's layout.
            shape = residual.shape
            normalized, summed = torch.ops.shapefuse.project_add_norm(
                hidden_states.reshape(-1, hidden_states.shape[-1]), self.projection.weight,
                residual.reshape(-1, residual.shape[-1]), self.post_attention_layernorm.weight,
                self.site.site_id)
            hidden_states = self.mlp(normalized.view(shape)) + summed.view(shape)
            return (hidden_states, weights) if output_attentions else (hidden_states,)
    model.model.layers = torch.nn.ModuleList([AdaptedLayer(layer) for layer in model.model.layers])
    return model


def run_pytorch(args):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM, StaticCache
    if not torch.cuda.is_available():
        raise RuntimeError("System experiment requires CUDA")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    config = Qwen3Config(**MODEL_CONFIG)
    config._attn_implementation = "sdpa"
    with torch.inference_mode():
        base = Qwen3ForCausalLM(config).eval().half().cuda()
    parameters = sum(p.numel() for p in base.parameters())
    write_json(out / "manifest.json", environment() | {"model": MODEL_CONFIG,
               "parameters": parameters, "pretrained": False, "teacher_forced": True,
               "tokens_generated_per_sequence": 64, "outer_compile": "default, cudagraphs=False",
               "projection_binding": "opaque FP16 linear, compiled add+RMSNorm, emulate_precision_casts=True"})
    write_json(out / "config.json", MODEL_CONFIG)
    cases = [(b, s) for b in (1, 8) for s in (128, 2048)]
    if args.case:
        cases = [(b, s) for b, s in cases if f"b{b}_s{s}" == args.case]
    if not cases:
        raise ValueError("Unknown model workload")
    policies = args.methods.split(",") if args.methods else POLICIES
    policies = list(policies)
    __import__("random").Random(args.seed).shuffle(policies)
    rows, details = [], []
    for b, s in cases:
        gen = torch.Generator().manual_seed(args.seed + b + s)
        tokens = torch.randint(3, config.vocab_size, (b, s + 64), generator=gen).cuda()
        references = None
        for policy in policies:
            if policy not in POLICIES:
                raise ValueError(policy)
            print("MODEL", b, s, policy, flush=True)
            record = dict(batch=b, prompt_length=s, policy=policy, decode_steps=64, seed=args.seed)
            try:
                model = copy.deepcopy(base)
                if policy != "original":
                    model = adapt_hf(model, policy)
                cache = StaticCache(config=config, max_batch_size=b, max_cache_len=s+64,
                                    device="cuda", dtype=torch.float16)
                attention_mask = torch.ones((b, s+64), device="cuda", dtype=torch.long)
                prepos = torch.arange(s, device="cuda")
                def prefill():
                    return model(tokens[:, :s], attention_mask=attention_mask, past_key_values=cache,
                                 cache_position=prepos, use_cache=True, logits_to_keep=1).logits
                def decode_steps(collect=False):
                    outputs = []
                    for i in range(64):
                        pos = torch.tensor([s+i], device="cuda")
                        y = model(tokens[:, s+i:s+i+1], attention_mask=attention_mask,
                                  past_key_values=cache, cache_position=pos, use_cache=True,
                                  logits_to_keep=1).logits
                        if collect:
                            outputs.append(y.detach().clone())
                    return outputs
                with torch.inference_mode():
                    setup = time.perf_counter()
                    eager_prefill = prefill().detach().clone()
                    eager_decode = decode_steps(collect=True)
                    torch.cuda.synchronize()
                    if references is None:
                        # A reference always comes from the unmodified model.
                        reference_model = base
                        saved_model = model
                        model = reference_model
                        cache.reset()
                        reference_prefill = prefill().detach().clone()
                        reference_decode = decode_steps(collect=True)
                        model = saved_model
                        references = [reference_prefill, *reference_decode]
                    actual = [eager_prefill, *eager_decode]
                    errors = [float((a-r).abs().max()) for a, r in zip(actual, references)]
                    passed = all(bool(torch.isfinite(a).all()) and torch.allclose(a, r, atol=1e-2, rtol=1e-2)
                                 for a, r in zip(actual, references))
                    if not passed:
                        raise ArithmeticError(f"Full-model logits failed: max_error={max(errors)}")
                    model = torch.compile(model, options={"triton.cudagraphs": False})
                    cache.reset()
                    compiled_prefill = prefill().detach().clone()
                    compiled_decode = decode_steps(collect=True)
                    for a, r in zip([compiled_prefill, *compiled_decode], references):
                        if not torch.isfinite(a).all() or not torch.allclose(a, r, atol=1e-2, rtol=1e-2):
                            raise ArithmeticError("Compiled full-model logits failed")
                    torch.cuda.synchronize()
                    record["setup_ms"] = (time.perf_counter()-setup)*1000
                    record["max_abs_logit_error"] = max(errors)
                    record["numerics_passed"] = True
                    premeasurement = measure(lambda: prefill(), (), warmup=args.warmup or 5,
                                             repeats=args.repeats or 20)
                    prefill()
                    decmeasurement = measure(lambda: decode_steps(), (), warmup=2,
                                             repeats=args.repeats or 20)
                    # The compiled wrapper exposes the original module here.
                    original = getattr(model, "_orig_mod", model)
                    sites = [layer.site for layer in original.model.layers if hasattr(layer, "site")]
                    record["selection"] = [dict(layer=i, **p) for i, site in enumerate(sites)
                                           for p in site.prepared]
                    record["calibration_ms"] = sum(p["calibration_ms"] for site in sites for p in site.prepared)
                    record["cache_hits"] = sum(site.executor.hits for site in sites)
                    record["cache_misses"] = sum(site.executor.misses for site in sites)
                    for phase, timing in (("prefill", premeasurement), ("decode64", decmeasurement)):
                        row = dict(batch=b, prompt_length=s, policy=policy, phase=phase, seed=args.seed,
                                   execution_status="ok", decode_steps=64,
                                   **{k:v for k,v in timing.items() if k != "gpu_samples_ms"})
                        row["tokens_per_second"] = b * (s if phase == "prefill" else 64) / (
                            timing["host_batch_ms_per_call"] / 1000)
                        rows.append(row)
                        record[phase] = timing
                record["execution_status"] = "ok"
                print("PASS", record["setup_ms"], flush=True)
            except Exception as exc:
                import traceback
                record.update(execution_status="failed", error=f"{type(exc).__name__}: {exc}",
                              traceback=traceback.format_exc())
                print(record["error"], flush=True)
                rows.append(dict(batch=b, prompt_length=s, policy=policy, phase="all",
                                 seed=args.seed, execution_status="failed", error=record["error"]))
            details.append(record)
            write_csv(out / "latency.csv", rows)
            write_json(out / "details.json", details)
            del model
            torch._dynamo.reset()
            torch.cuda.empty_cache()
    write_json(out / "completed.json", {"rows":len(rows), "ok":sum(r["execution_status"]=="ok" for r in rows)})
    return 0
