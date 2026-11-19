# TPCA reimplementation

This is a practical Python reimplementation of the algorithms described in the paper **Probabilistic PCA on tensors** (arXiv:2510.19516).

Files:
- `tpca_reimplementation.py` — model code for the EM algorithm and the rank-1 power iteration.
- `run_tpca_experiments.py` — small runner for the three numerical cases described in Section 6.

## What is implemented

### EM algorithm
- E-step from equations (15)–(18)
- M-step objective from equation (19)
- noise update from equation (21)
- closed-form mode-wise factor update from Appendix B, equation (40)
- HOSVD warm start used in the experiments

### Rank-1 power iteration
- update rule from equations (25)–(30)
- a scalable implementation that avoids explicitly materializing the full paired covariance tensor
- signal-strength estimate `omega_hat` and residual noise estimate `sigma2_hat`

## Important caveat

The paper does not appear to provide public code. This implementation is therefore a **faithful re-creation from the paper text**, not the authors' original code. The formulas are respected, but some engineering details had to be chosen explicitly, such as:
- exact unfolding conventions in NumPy
- HOSVD initialization details
- stopping criterion details
- whether to center the data explicitly in the zero-mean experiments

For the experiments in Section 6, the code follows the paper's zero-mean setting (`nu = 0`) by default.

## Example usage

### Case 1: EM algorithm

```bash
PYTHONPATH=/mnt/data python /mnt/data/run_tpca_experiments.py \
  --case case1 --n 5 --m 2 --sigma2 1.0 --N 15 --replications 3
```

### Case 2: power iteration convergence

```bash
PYTHONPATH=/mnt/data python /mnt/data/run_tpca_experiments.py \
  --case case2 --L 5 --replications 3
```

### Case 3: power iteration, effect of `N` and `sigma^2`

```bash
PYTHONPATH=/mnt/data python /mnt/data/run_tpca_experiments.py \
  --case case3 --sigma2 4 --N 200 --L 5 --replications 3
```
