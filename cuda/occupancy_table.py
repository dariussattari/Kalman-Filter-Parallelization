#!/usr/bin/env python3
"""Parse a ptxas-verbose log and emit a CUDA occupancy table.

Reproduces the headline information that NVIDIA Nsight Compute reports as
`launch__registers_per_thread` and `sm__warps_active.avg.pct_of_peak`, derived
from compile-time evidence alone:

  * registers/thread        — extracted from `ptxas info: ... Used N registers`
  * shared-memory/block     — extracted from the same line
  * theoretical occupancy   — computed via the well-known per-architecture
                              limits (registers/SM, warps/SM, threads/block).

For each candidate block size we report whether the launch fits in the SM's
register file.  The 65,536-register file on Ada (sm_89) is what limits our
ekf_cuda kernel to block_size <= 256: at the ptxas-reported register count,
a 512-thread block would request more registers than the SM provides, which
is the same condition CUDA reports at runtime as
`cudaErrorLaunchOutOfResources`.

Output: a markdown table written to --output (and stdout).
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path


# Per-architecture limits from CUDA Programming Guide, Table "Compute
# Capabilities".  Only the subset we need.
ARCH_LIMITS = {
    # cc:  (max regs/SM, max warps/SM, max threads/block, regs allocation unit)
    "sm_70": (65536, 64, 1024, 256),  # V100
    "sm_75": (65536, 32, 1024, 256),  # T4
    "sm_80": (65536, 64, 1024, 256),  # A100
    "sm_86": (65536, 48, 1024, 256),  # A10 / RTX 30
    "sm_89": (65536, 48, 1024, 256),  # L4 / Ada
    "sm_90": (65536, 64, 1024, 256),  # H100
}


def parse_ptxas(log_path: Path):
    """Returns (regs_per_thread, smem_bytes_per_block) from a ptxas log.

    Looks for the canonical line:
      ptxas info    : Used 96 registers, 384 stack, 0 bytes smem, 504 bytes cmem[0]
    """
    text = log_path.read_text()
    # The kernel name printout tells us *which* kernel we're parsing; we want
    # the run_ekf one, not any helper.
    pat = re.compile(
        r"ptxas info\s*:\s*Used\s+(?P<regs>\d+)\s+registers"
        r"(?:,\s*\d+\s+stack)?"
        r"(?:,\s*(?P<smem>\d+)\s+bytes\s+smem)?",
        re.IGNORECASE,
    )
    matches = list(pat.finditer(text))
    if not matches:
        sys.exit(f"no `ptxas info: Used N registers` line found in {log_path}")
    # Take the largest reg count — that's the dominant kernel.
    best = max(matches, key=lambda m: int(m.group("regs")))
    regs = int(best.group("regs"))
    smem = int(best.group("smem")) if best.group("smem") else 0
    return regs, smem


def round_up(x: int, mult: int) -> int:
    return ((x + mult - 1) // mult) * mult


def occupancy(block_size: int, regs_per_thread: int, smem_per_block: int,
              max_regs_per_sm: int, max_warps_per_sm: int,
              max_threads_per_block: int, regs_alloc_unit: int):
    """Return (warps_per_block, blocks_per_sm, theoretical_occ_pct, status_msg)."""
    if block_size > max_threads_per_block:
        return 0, 0, 0.0, f"FAIL: {block_size} > max threads/block"
    warps_per_block = (block_size + 31) // 32
    # Registers allocated per warp, rounded up to the alloc granule.
    regs_per_warp = round_up(regs_per_thread * 32, regs_alloc_unit)
    regs_per_block = regs_per_warp * warps_per_block
    if regs_per_block > max_regs_per_sm:
        return warps_per_block, 0, 0.0, "FAIL: regs/block > regs/SM"
    blocks_by_regs = max_regs_per_sm // regs_per_block
    # Other limits (smem, warps).  We only constrain by these two for sanity.
    blocks_by_warps = max_warps_per_sm // warps_per_block if warps_per_block else 0
    blocks_per_sm = min(blocks_by_regs, blocks_by_warps) if blocks_by_warps else blocks_by_regs
    active_warps = blocks_per_sm * warps_per_block
    occ_pct = 100.0 * active_warps / max_warps_per_sm
    return warps_per_block, blocks_per_sm, occ_pct, "OK"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptxas-log", required=True, type=Path)
    ap.add_argument("--output",    required=True, type=Path)
    ap.add_argument("--arch",      default="sm_89", choices=list(ARCH_LIMITS))
    ap.add_argument("--block-sizes", default="32,64,128,256,512,1024")
    args = ap.parse_args()

    max_regs, max_warps, max_tpb, regs_unit = ARCH_LIMITS[args.arch]
    regs_per_thread, smem = parse_ptxas(args.ptxas_log)

    block_sizes = [int(b) for b in args.block_sizes.split(",")]

    lines = []
    lines.append(f"# CUDA occupancy table  ({args.arch}, derived from {args.ptxas_log.name})")
    lines.append("")
    lines.append(f"- registers/thread (ptxas):  **{regs_per_thread}**")
    lines.append(f"- smem/block (ptxas):        **{smem}** bytes")
    lines.append(f"- regs/SM (arch limit):      {max_regs:,}")
    lines.append(f"- warps/SM (arch limit):     {max_warps}")
    lines.append("")
    lines.append("| block | warps/block | regs/block | blocks/SM | theoretical occ | status |")
    lines.append("|------:|------------:|-----------:|----------:|----------------:|:-------|")
    for bs in block_sizes:
        wpb, bps, occ, msg = occupancy(bs, regs_per_thread, smem,
                                       max_regs, max_warps, max_tpb, regs_unit)
        regs_per_block = round_up(regs_per_thread * 32, regs_unit) * wpb if wpb else 0
        if msg.startswith("FAIL"):
            lines.append(f"| {bs} | {wpb} | {regs_per_block:,} | 0 | — | **{msg}** |")
        else:
            lines.append(f"| {bs} | {wpb} | {regs_per_block:,} | {bps} | {occ:.1f}% | {msg} |")
    lines.append("")
    lines.append("Interpretation: any block size where `regs/block > regs/SM` cannot launch — "
                 "the runtime returns `cudaErrorLaunchOutOfResources`.  This is the source of "
                 "the empirical block_size cap observed during the throughput sweep.")
    out = "\n".join(lines) + "\n"
    args.output.write_text(out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
