"""Datasheet Roofline anchors.

Peaks are public dense Tensor-Core numbers, not measurements from the
machine that runs this repo. Sparsity (2x) peaks are intentionally unused.
Replace ``peak_flops_per_s`` and ``peak_bytes_per_s`` with measured values
before quoting a ridge in the paper; until then every ridge is a datasheet
anchor and measured bandwidth stays null.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Machine:
    """One peak-FLOP/s and peak-byte/s pair, named by the dtype of that peak."""

    name: str
    peak_flops_per_s: float
    peak_bytes_per_s: float
    peak_dtype: str
    source: str

    @property
    def ridge(self) -> float:
        """Balance point R = peak FLOP/s / peak byte/s, in FLOP/byte."""
        return self.peak_flops_per_s / self.peak_bytes_per_s

    @property
    def peak_tflops(self) -> float:
        return self.peak_flops_per_s / 1e12

    @property
    def peak_tb_per_s(self) -> float:
        return self.peak_bytes_per_s / 1e12


# H100 SXM: dense FP16 Tensor Core 989 TFLOP/s, HBM3 3.35 TB/s.
H100_SXM_FP16 = Machine(
    name="H100_SXM_FP16",
    peak_flops_per_s=989e12,
    peak_bytes_per_s=3.35e12,
    peak_dtype="fp16",
    source="NVIDIA H100 SXM datasheet, dense FP16 Tensor Core, HBM3 3.35 TB/s",
)

# H100 SXM: dense FP8 Tensor Core 1979 TFLOP/s, same HBM.
H100_SXM_FP8 = Machine(
    name="H100_SXM_FP8",
    peak_flops_per_s=1979e12,
    peak_bytes_per_s=3.35e12,
    peak_dtype="fp8",
    source="NVIDIA H100 SXM datasheet, dense FP8 Tensor Core, HBM3 3.35 TB/s",
)

# A100 80GB SXM: dense FP16 Tensor Core 312 TFLOP/s, HBM2e 2039 GB/s.
A100_SXM80_FP16 = Machine(
    name="A100_SXM80_FP16",
    peak_flops_per_s=312e12,
    peak_bytes_per_s=2.039e12,
    peak_dtype="fp16",
    source="NVIDIA A100 80GB SXM datasheet, dense FP16 Tensor Core, HBM2e 2039 GB/s",
)

# A40: dense FP16 Tensor Core 149.7 TFLOP/s (sparsity 299.4 is not used),
# GDDR6 696 GB/s. Datasheet, not a measured ridge from any run in this repo.
A40_FP16 = Machine(
    name="A40_FP16",
    peak_flops_per_s=149.7e12,
    peak_bytes_per_s=696e9,
    peak_dtype="fp16",
    source=(
        "NVIDIA A40 datasheet, dense FP16 Tensor Core with FP16 accumulate "
        "149.7 TFLOP/s (structural sparsity 299.4 unused), GDDR6 696 GB/s"
    ),
)

MACHINES: dict[str, Machine] = {
    H100_SXM_FP16.name: H100_SXM_FP16,
    H100_SXM_FP8.name: H100_SXM_FP8,
    A100_SXM80_FP16.name: A100_SXM80_FP16,
    A40_FP16.name: A40_FP16,
}

# Short names for the CLI. They resolve to the datasheet anchors above.
ALIASES: dict[str, str] = {
    "h100": H100_SXM_FP16.name,
    "h100_fp16": H100_SXM_FP16.name,
    "h100_fp8": H100_SXM_FP8.name,
    "a100": A100_SXM80_FP16.name,
    "a100_fp16": A100_SXM80_FP16.name,
    "a40": A40_FP16.name,
    "a40_fp16": A40_FP16.name,
}


def get_machine(name: str) -> Machine:
    key = name.strip()
    if key in MACHINES:
        return MACHINES[key]
    alias = ALIASES.get(key) or ALIASES.get(key.lower())
    if alias is not None:
        return MACHINES[alias]
    known = ", ".join(sorted(MACHINES))
    aliases = ", ".join(sorted(ALIASES))
    raise KeyError(f"Unknown machine {name!r}. Known: {known}. Aliases: {aliases}")
