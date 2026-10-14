#!/usr/bin/env python3
"""Validate whole benchmark outputs and diagnose the legacy FP16 RMSNorm.

Uses the harness's tensor generators and operation definitions. GEMM is
checked against FP32 normalization of the FP16 product. Attention paths
are compared against materialized attention with FP32 intermediates.
This is numerical validation, not measured DRAM classifier accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch.nn.functional as F
    import torch._dynamo
    from shapefuse.bakeoff import _torch_rmsnorm

    if not torch.cuda.is_available():
        parser.error("Numerical validation requires CUDA")
    torch._dynamo.config.cache_size_limit = 64
    rows = []

    def inspect(output, reference, rtol, atol):
        finite = bool(torch.isfinite(output).all())
        close = finite and bool(torch.allclose(output, reference, rtol=rtol, atol=atol))
        return {"finite": finite, "close": close,
                "max_abs_error": float((output - reference).abs().max()),
                "zero_rows": int((output.abs().sum(dim=-1) == 0).sum())}

    with torch.no_grad():
        for m in (1, 32, 128, 256, 512, 2048, 4096):
            g = torch.Generator(device="cpu").manual_seed(0)
            a = torch.randn(m, 4096, generator=g).to(device="cuda", dtype=torch.float16)
            b = torch.randn(4096, 4096, generator=g).to(device="cuda", dtype=torch.float16)
            weight = torch.ones(4096, device="cuda", dtype=torch.float16)

            def chain():
                return _torch_rmsnorm(a @ b, weight)

            def legacy_chain():
                c = a @ b
                var = (c * c).mean(dim=-1, keepdim=True)
                return c * torch.rsqrt(var + 1e-6) * weight

            c = (a @ b).float()
            reference = (c * torch.rsqrt(c.square().mean(-1, keepdim=True) + 1e-6)).half()
            eager, compiled = chain(), torch.compile(chain, dynamic=False)()
            legacy, legacy_compiled = legacy_chain(), torch.compile(legacy_chain, dynamic=False)()
            row = {"op": "matmul_rmsnorm", "M": m, "rtol": 1e-3, "atol": 1e-3,
                   "eager": inspect(eager, reference, 1e-3, 1e-3),
                   "compiled": inspect(compiled, reference, 1e-3, 1e-3),
                   "legacy_eager": inspect(legacy, reference, 1e-3, 1e-3),
                   "legacy_compiled": inspect(legacy_compiled, reference, 1e-3, 1e-3),
                   "legacy_eager_compiled_max_abs": float((legacy - legacy_compiled).abs().max())}
            rows.append(row)
            print(row, flush=True)

        for sq, sk in ((128,128),(512,512),(2048,2048),(1,128),(1,2048)):
            g = torch.Generator(device="cpu").manual_seed(0)
            q = torch.randn(1,8,sq,64,generator=g).to(device="cuda",dtype=torch.float16)
            k = torch.randn(1,8,sk,64,generator=g).to(device="cuda",dtype=torch.float16)
            v = torch.randn(1,8,sk,64,generator=g).to(device="cuda",dtype=torch.float16)
            weight = torch.ones(64, device="cuda", dtype=torch.float16)
            def materialized():
                scores = (q @ k.transpose(-1,-2)) / 8
                return _torch_rmsnorm(torch.softmax(scores,dim=-1) @ v,weight)
            def sdpa_chain():
                return _torch_rmsnorm(F.scaled_dot_product_attention(q,k,v),weight)
            scores = (q.float() @ k.float().transpose(-1,-2)) / 8
            body = torch.softmax(scores,dim=-1) @ v.float()
            reference = (body * torch.rsqrt(body.square().mean(-1,keepdim=True) + 1e-6)).half()
            row = {"op": "attention", "query_length": sq, "cache_length": sk,
                   "rtol": 1e-2, "atol": 1e-2,
                   "materialized": inspect(materialized(),reference,1e-2,1e-2),
                   "sdpa": inspect(sdpa_chain(),reference,1e-2,1e-2),
                   "compiled": inspect(torch.compile(sdpa_chain,dynamic=False)(),reference,1e-2,1e-2)}
            rows.append(row)
            print(row, flush=True)
    passed = all(row[name]["close"] for row in rows for name in
                 (("eager","compiled") if row["op"] == "matmul_rmsnorm"
                  else ("materialized","sdpa","compiled")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"device":torch.cuda.get_device_name(),
                                   "torch_version":torch.__version__,
                                   "cuda_version":torch.version.cuda,
                                   "source_sha256":hashlib.sha256((REPO / "shapefuse/bakeoff.py").read_bytes()).hexdigest(),
                                   "seed":0,"passed":passed,"rows":rows},indent=2)+"\n",
                        encoding="utf-8")
    if not passed:
        raise SystemExit("Corrected paths failed numerical validation")


if __name__ == "__main__":
    main()
