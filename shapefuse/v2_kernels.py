"""Compiler-visible FP16 projection/residual/RMSNorm candidates (opt-in).

HF rounds the residual before normalization; vLLM normalizes the FP32 sum.
No input mutation in these functional candidates; serving mutation is separate.
"""
import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

VERSION = 'scic-v2-kernels-1'

@triton.jit
def _partial(X, W, P, M, N: tl.constexpr, K: tl.constexpr,
             SPLIT: tl.constexpr, BM: tl.constexpr=16, BN: tl.constexpr=64,
             BK: tl.constexpr=64):
    mi = tl.program_id(0)*BM + tl.arange(0, BM)
    ni = tl.program_id(1)*BN + tl.arange(0, BN)
    si = tl.program_id(2)
    tiles: tl.constexpr = triton.cdiv(K, BK*SPLIT)
    acc = tl.full((BM, BN), 0, tl.float32)
    for ki in range(tiles):
        kk = (si*tiles+ki)*BK + tl.arange(0, BK)
        a = tl.load(X+mi[:, None]*K+kk[None, :], (mi[:, None]<M)&(kk[None, :]<K), 0)
        b = tl.load(W+ni[None, :]*K+kk[:, None], (ni[None, :]<N)&(kk[:, None]<K), 0)
        acc = tl.dot(a, b, acc)
    tl.store(P+si*M*N+mi[:, None]*N+ni[None, :], acc,
             (mi[:, None]<M)&(ni[None, :]<N))

@triton.jit
def _finish(C, R, G, Y, S, M, N: tl.constexpr,
            SPLIT: tl.constexpr, PARTIAL: tl.constexpr, VLLM: tl.constexpr,
            EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    ns = tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for i in range(SPLIT):
        acc += tl.load(C+i*M*N+row*N+ns, ns<N, 0).to(tl.float32)
    product = acc.to(tl.float16).to(tl.float32)
    rv = tl.load(R+row*N+ns, ns<N, 0).to(tl.float32)
    val = product+rv
    summed = val.to(tl.float16)
    if not VLLM:
        val = summed.to(tl.float32)
    variance = tl.sum(tl.where(ns<N, val*val, 0), 0)/N
    norm = val*tl.rsqrt(variance+EPS)
    gamma = tl.load(G+ns, ns<N, 0).to(tl.float32)
    if not VLLM:
        norm = norm.to(tl.float16).to(tl.float32)
    tl.store(Y+row*N+ns, norm*gamma, ns<N)
    tl.store(S+row*N+ns, summed, ns<N)

@triton_op('shapefuse_v2::blas_norm', mutates_args={})
def blas_norm(x: torch.Tensor, w: torch.Tensor, r: torch.Tensor, g: torch.Tensor,
              warps: int, vllm: bool, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    c = torch.nn.functional.linear(x, w)
    y, s = torch.empty_like(r), torch.empty_like(r)
    m, n = r.numel()//r.shape[-1], r.shape[-1]
    wrap_triton(_finish)[(m,)](c,r,g,y,s,m,n,1,False,vllm,eps,
                               triton.next_power_of_2(n),num_warps=warps)
    return y,s

@triton_op('shapefuse_v2::split_norm', mutates_args={})
def split_norm(x: torch.Tensor, w: torch.Tensor, r: torch.Tensor, g: torch.Tensor,
               split: int, vllm: bool, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    m,n,k = r.numel()//r.shape[-1],w.shape[0],w.shape[1]
    p = torch.empty((split,m,n),device=x.device,dtype=torch.float32)
    y,s = torch.empty_like(r),torch.empty_like(r)
    wrap_triton(_partial)[(triton.cdiv(m,16),triton.cdiv(n,64),split)](x,w,p,m,n,k,split,16,64,64,num_warps=4)
    wrap_triton(_finish)[(m,)](p,r,g,y,s,m,n,split,True,vllm,eps,
                               triton.next_power_of_2(n),num_warps=4)
    return y,s

NAMES = ('split1','split4','split8','blas4','blas8')

def reference(x,w,r,g,*,vllm=False,eps=1e-6):
    c = torch.nn.functional.linear(x,w)
    val = c.float()+r.float() if vllm else (c+r).float()
    norm = val*torch.rsqrt(val.square().mean(-1,keepdim=True)+eps)
    y = (norm*g.float()).half() if vllm else norm.half()*g
    return y,val.half()


def candidate(name, *, vllm=False, eps=1e-6):
    if name.startswith('split'):
        split=int(name[5:])
        return lambda x,w,r,g: split_norm(x,w,r,g,split,vllm,eps)
    if name.startswith('blas'):
        warps=int(name[4:])
        return lambda x,w,r,g: blas_norm(x,w,r,g,warps,vllm,eps)
    if name == 'native':
        return lambda x,w,r,g: reference(x,w,r,g,vllm=vllm,eps=eps)
    raise ValueError(name)

@triton_op('shapefuse_v2::serving_norm', mutates_args={'r'})
def serving_norm(x: torch.Tensor, w: torch.Tensor, r: torch.Tensor, g: torch.Tensor,
                 split: int, warps: int, eps: float) -> torch.Tensor:
    """vLLM contract: update residual in-place, return fresh normalized storage."""
    m,n,k=r.numel()//r.shape[-1],w.shape[0],w.shape[1]
    y=torch.empty_like(r)
    if split:
        c=torch.empty((split,m,n),device=x.device,dtype=torch.float32)
        wrap_triton(_partial)[(triton.cdiv(m,16),triton.cdiv(n,64),split)](x,w,c,m,n,k,split,16,64,64,num_warps=4)
        parts=split
    else:
        c=torch.nn.functional.linear(x,w);parts=1
    wrap_triton(_finish)[(m,)](c,r,g,y,r,m,n,parts,bool(split),True,eps,triton.next_power_of_2(n),num_warps=warps)
    return y
