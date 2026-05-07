# CUDA occupancy table  (sm_89, derived from ptxas.log)

- registers/thread (ptxas):  **255**
- smem/block (ptxas):        **0** bytes
- regs/SM (arch limit):      65,536
- warps/SM (arch limit):     48

| block | warps/block | regs/block | blocks/SM | theoretical occ | status |
|------:|------------:|-----------:|----------:|----------------:|:-------|
| 32 | 1 | 8,192 | 8 | 16.7% | OK |
| 64 | 2 | 16,384 | 4 | 16.7% | OK |
| 128 | 4 | 32,768 | 2 | 16.7% | OK |
| 256 | 8 | 65,536 | 1 | 16.7% | OK |
| 512 | 16 | 131,072 | 0 | — | **FAIL: regs/block > regs/SM** |
| 1024 | 32 | 262,144 | 0 | — | **FAIL: regs/block > regs/SM** |

Interpretation: any block size where `regs/block > regs/SM` cannot launch — the runtime returns `cudaErrorLaunchOutOfResources`.  This is the source of the empirical block_size cap observed during the throughput sweep.
