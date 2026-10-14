"""Adapters execute official research schedules, without replacing their algorithms."""
from __future__ import annotations
import importlib.util
import math
import os
from pathlib import Path
import sys

from shapefuse.bakeoff import _torch_rmsnorm


def neptune_factory(case, *, torch_bridge=False):
    import torch
    import tvm
    from tvm import te, tir
    from tvm.neptune import tvm_triton_build
    root=Path(os.environ["NEPTUNE_ROOT"])
    sys.path.insert(0,str(root))
    from scripts.neptune_bench.ours.attn_plain import (
        create_general_attention, schedule_full_attn_flash, schedule_mask_attn_flash,
        schedule_flash_decoding, causal_mask_cond)
    if case["H"]!=case["Hkv"]:
        # This adapter currently covers the official plain-attention schedules.
        # GQA needs a separately reproduced official GQA schedule.
        raise NotImplementedError("Plain Neptune adapter does not support GQA")
    shape=tuple(case[k] for k in ("B","H","Sq","Sk","D"))
    mod=create_general_attention(shape,mask_cond=causal_mask_cond if case["causal"] else None)
    sch=tir.Schedule(mod)
    sch.work_on(next(iter(mod.functions)).name_hint)
    scheduler=schedule_flash_decoding if case["Sq"]==1 else (
        schedule_mask_attn_flash if case["causal"] else schedule_full_attn_flash)
    sch=scheduler(sch)
    target=tvm.target.Target(f"cuda -arch=sm_86",host="llvm")
    if torch_bridge:
        from tvm.neptune import extract_triton_kernels_as_torch_fns
        kernels=extract_triton_kernels_as_torch_fns(sch.mod,target)
        if len(kernels)!=1:
            raise NotImplementedError("Official torch bridge requires a single attention kernel")
        kernel=kernels[0]
    else:
        kernel=tvm_triton_build(sch.mod,target)
    from shapefuse.supplement import compiled
    norm=compiled(_torch_rmsnorm,"default")
    def run(q,k,v,w):
        # DLPack bridges the live CUDA tensors without CPU copies. Adopt the
        # current stream for runtime launches and time the complete chain.
        result=torch.empty_like(q)
        if torch_bridge:
            kernel(q,k,v,result)
        else:
            tvm.cuda(0).set_raw_stream(torch.cuda.current_stream().cuda_stream)
            arrays=[tvm.nd.from_dlpack(torch.utils.dlpack.to_dlpack(x)) for x in (q,k,v,result)]
            kernel(*arrays)
        return norm(result,w)
    return run


def generate_redfuser(case):
    import tvm
    from tvm import te
    if case["causal"] or case["H"]!=case["Hkv"]:
        raise NotImplementedError("Official RedFuser flash-attention example is unmasked MHA")
    # The shipped example has one fixed shape. Its generator is invoked with
    # only placeholder sizes changed; all official transformation passes stay.
    from tvm.redfuser.example import flash_attention as example
    original=te.placeholder
    shapes={"q":(case["B"],case["H"],case["Sq"],case["D"]),
            "k":(case["B"],case["H"],case["Sk"],case["D"]),
            "v":(case["B"],case["H"],case["Sk"],case["D"])}
    def placeholder(shape,*args,**kwargs):
        name=kwargs.get("name", args[1] if len(args)>1 else None)
        return original(shapes.get(name,shape),*args,**kwargs)
    name=f"supplement_{case['label']}"
    folder=Path(example.__file__).parent/"generated"
    folder.mkdir(exist_ok=True)
    try:
        te.placeholder=placeholder
        example.main(name,{"q_len":64,"kv_len":64,"head_dim_qk":case["D"],"head_dim_v":case["D"]})
    finally:
        te.placeholder=original
    path=folder/f"generated_{name}.py"
    return str(path)


def redfuser_factory(case):
    import json
    import subprocess
    if case["causal"] or case["H"]!=case["Hkv"]:
        raise NotImplementedError("Official RedFuser flash-attention example is unmasked MHA")
    env=os.environ.copy()
    env["PYTHONPATH"]=os.environ["REDFUSER_GENERATOR_PATH"]
    code="import json,sys; from shapefuse.research import generate_redfuser; print('OUTPUT_PATH='+generate_redfuser(json.loads(sys.argv[1])))"
    result=subprocess.run([sys.executable,"-c",code,json.dumps(case)],env=env,
                          text=True,capture_output=True,timeout=120)
    if result.returncode:
        raise RuntimeError(result.stderr[-5000:])
    name=f"supplement_{case['label']}"
    path=next(line.split("=",1)[1] for line in result.stdout.splitlines() if line.startswith("OUTPUT_PATH="))
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel=getattr(module,name)()
    from shapefuse.supplement import compiled
    norm=compiled(_torch_rmsnorm,"default")
    def run(q,k,v,w):
        return norm(kernel(q,k,v),w)
    return run


def research_recipes(case):
    return {"neptune":lambda:neptune_factory(case),
            "neptune_torch":lambda:neptune_factory(case,torch_bridge=True),
            "redfuser":lambda:redfuser_factory(case),
            "redfuser_shipped":lambda:redfuser_shipped_factory(case)}


def redfuser_shipped_factory(case):
    """Parameterize only literal shapes/grid in the artifact's committed kernel.

    This is reported separately from regeneration, whose numerical gate failed.
    It does not establish successful automatic regeneration by RedFuser.
    """
    import re
    import tempfile
    if case["causal"] or case["H"]!=case["Hkv"] or case["D"]!=64:
        raise NotImplementedError("Shipped RedFuser artifact: unmasked MHA, D=64")
    root=Path(os.environ["REDFUSER_ROOT"])
    source=root/"python/tvm/redfuser/example/generated/generated_redfuser_flash_attention.py"
    code=source.read_text()
    b,h,sq,sk=tuple(case[k] for k in ("B","H","Sq","Sk"))
    for name,seq in (("q",sq),("k",sk),("v",sk),("Cast",sq)):
        old=f'{name}: T.Tensor([128, 16, 512, 64], "float16")'
        new=f'{name}: T.Tensor([{b}, {h}, {seq}, 64], "float16")'
        if old not in code:
            raise RuntimeError("Unexpected committed artifact shape; inspect source")
        code=code.replace(old,new)
    code=code.replace("T.Kernel(8, 16, 128)",f"T.Kernel({math.ceil(sq/64)}, {h}, {b})")
    code=code.replace("T.Pipelined(0, 8,",f"T.Pipelined(0, {math.ceil(sk/64)},")
    folder=Path(os.environ.get("REDFUSER_ADAPTED_DIR","/root/test/.experiments/redfuser_shipped"))
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/(case["label"]+".py")
    path.write_text(code)
    spec=importlib.util.spec_from_file_location("redfuser_shipped_"+case["label"],path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    kernel=module.redfuser_flash_attention()
    from shapefuse.supplement import compiled
    norm=compiled(_torch_rmsnorm,"default")
    def run(q,k,v,w):
        return norm(kernel(q,k,v),w)
    return run
