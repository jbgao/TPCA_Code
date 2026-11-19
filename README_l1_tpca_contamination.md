# L1-TPCA contamination experiments

This adds a more informative benchmark for the current L1-TPCA prototype.

## Why this script helps

A pure isotropic Laplace-noise experiment is **not** the strongest place to show the benefit of the robust model, because Gaussian TPCA still sees the correct second-order covariance form.

This script focuses on settings where robustness is more meaningful:
- `tensor_outliers`: a small subset of whole tensors receives very large additive noise
- `slice_outliers`: a small subset of tensors has one corrupted slice
- `hybrid`: both of the above

It also includes two reference settings:
- `gaussian_clean`
- `laplace_clean`

## Files

- `l1_tpca_contamination.py`: data-generation and evaluation helpers
- `run_l1_tpca_contamination.py`: command-line driver

## Example commands

Whole-tensor contamination:

```bash
PYTHONPATH=/mnt/data python /mnt/data/run_l1_tpca_contamination.py \
  --scenario tensor_outliers --shape 6,6,6 --ranks 2,2,2 --N 60 \
  --sigma2 1.0 --contamination_rate 0.1 --outlier_scale 8.0 \
  --replications 10 --vbem_iter 40 --power_iter 10 --gaussian_iter 40 --device auto
```

Slice contamination:

```bash
PYTHONPATH=/mnt/data python /mnt/data/run_l1_tpca_contamination.py \
  --scenario slice_outliers --shape 6,6,6 --ranks 2,2,2 --N 60 \
  --sigma2 1.0 --contamination_rate 0.1 --outlier_scale 8.0 --slice_mode 1 \
  --replications 10 --device auto
```

Matched Laplace-noise baseline:

```bash
PYTHONPATH=/mnt/data python /mnt/data/run_l1_tpca_contamination.py \
  --scenario laplace_clean --shape 6,6,6 --ranks 2,2,2 --N 60 \
  --sigma2 1.0 --replications 10 --device auto
```

## What to look at

In the JSON output:
- `summary.robust_vbem.Err_mean` vs `summary.gaussian_em.Err_mean`
- `summary.robust_vbem.outlier_to_clean_weight_ratio_mean`
- `summary.robust_vbem.topk_low_weight_precision_mean`
- `summary.win_rates`

If robustness is helping, contaminated samples should get **smaller weights**, so the outlier-to-clean weight ratio should be noticeably below 1.
