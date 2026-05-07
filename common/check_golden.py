#!/usr/bin/env python3
"""Cross-implementation validation: compare a candidate per-realization
checksum CSV (m, checksum) against a reference (typically serial's golden.csv).

Exit code 0 = PASS (within tolerance); 1 = FAIL.

Usage:
    python3 check_golden.py --reference src/serial/results/golden.csv \\
                            --candidate src/openmp/results/golden.csv \\
                            --tol 1e-9
"""
from __future__ import annotations
import argparse
import sys
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reference", required=True, help="reference golden CSV (e.g. serial)")
    ap.add_argument("--candidate", required=True, help="candidate golden CSV (e.g. openmp)")
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="max allowed relative error (default 1e-9)")
    args = ap.parse_args()

    ref = np.loadtxt(args.reference, delimiter=",", skiprows=1)
    cand = np.loadtxt(args.candidate, delimiter=",", skiprows=1)

    if ref.ndim != 2 or ref.shape[1] != 2:
        sys.exit(f"reference {args.reference} has unexpected shape {ref.shape}")
    if cand.shape != ref.shape:
        print(f"FAIL: shape mismatch  ref {ref.shape}  cand {cand.shape}")
        return 1

    m_match = np.array_equal(ref[:, 0], cand[:, 0])
    diff = np.abs(ref[:, 1] - cand[:, 1])
    denom = np.maximum(np.abs(ref[:, 1]), 1e-300)
    rel = diff / denom
    worst_idx = int(np.argmax(rel))
    worst_rel = float(rel[worst_idx])
    worst_abs = float(diff.max())

    print(f"reference   : {args.reference}")
    print(f"candidate   : {args.candidate}")
    print(f"M           : {len(ref)}  (m-indices match: {m_match})")
    print(f"max abs diff: {worst_abs:.3e}")
    print(f"max rel diff: {worst_rel:.3e}  at m={int(ref[worst_idx, 0])}")
    print(f"tolerance   : {args.tol:.0e}")

    if not m_match or worst_rel > args.tol:
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
