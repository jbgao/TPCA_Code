# TPCA reimplementation with PyTorch

This is a PyTorch rewrite of the earlier NumPy/SciPy TPCA reimplementation.

Files:
- `tpca_reimplementation_torch.py` — TPCA EM and rank-1 power iteration with a PyTorch backend.
- `run_tpca_experiments_torch.py` — small runner for the three numerical cases from the paper.

## Design choices

- The public API mirrors the NumPy version as closely as possible.
- Internally, the main linear algebra now uses PyTorch.
- Results are returned as NumPy arrays and Python floats for easier downstream use.
- The code defaults to `float64` for numerical stability.
- Device selection supports `auto`, `cpu`, `cuda`, `cuda:0`, etc.

## What PyTorch is doing here

PyTorch is used for:
- tensor contractions and unfold/fold operations,
- Kronecker products,
- linear solves and matrix inverses,
- eigendecompositions and SVDs,
- CPU/GPU device management.

This makes it easier to move the code to GPU when the local environment has CUDA-enabled PyTorch.

## Important note

The execution environment used to generate this file had a CPU-only PyTorch install, so I could only test the CPU path here. The code is written to support CUDA devices when available in the user's own environment.

## Example usage

### Case 1: EM algorithm

```bash
PYTHONPATH=/mnt/data python /mnt/data/run_tpca_experiments_torch.py \
  --case case1 --n 5 --m 2 --sigma2 1.0 --N 15 --replications 3 --device auto
```

### Case 2: power iteration convergence

```bash
PYTHONPATH=/mnt/data python /mnt/data/run_tpca_experiments_torch.py \
  --case case2 --L 5 --replications 3 --device auto
```

### Case 3: power iteration, effect of `N` and `sigma^2`

```bash
PYTHONPATH=/mnt/data python /mnt/data/run_tpca_experiments_torch.py \
  --case case3 --sigma2 4 --N 200 --L 5 --replications 3 --device auto
```
