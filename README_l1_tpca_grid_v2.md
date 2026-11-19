# Grid runner for L1-TPCA version 2

This script runs many combinations of Version-2 contamination experiments and writes:

- a **raw CSV** with **one test run per row**
- a **summary CSV** aggregated over replications for each hyperparameter configuration
- a small **manifest JSON** describing the grid

## Files

- `run_l1_tpca_grid_v2.py`
- output: `<prefix>_raw.csv`
- output: `<prefix>_summary.csv`
- output: `<prefix>_manifest.json`

## Default grid

The default grid explores:

- `scenario in {gaussian_clean, laplace_clean, tensor_outliers, slice_outliers, hybrid}`
- `update_lam in {False, True}`
- `z_prior_precision in {1.0, 5.0, 10.0, 50.0}`
- `prior_scheme in {none, mild, strong}` for `update_lam=True`
- `mstep_inner = 5`
- `weight_damping = 0.0`
- `run_two_stage = True`
- `two_stage_blend = 1.0`

When `update_lam=False`, the runner automatically uses `prior_scheme=fixed` and sets
`lam_prior_a = lam_prior_b = 0`.

The prior schemes are:

- `none`: `(lam_prior_a, lam_prior_b) = (0, 0)`
- `mild`: `(5, 5 / lam_init)`
- `strong`: `(20, 20 / lam_init)`
- `very_strong`: `(50, 50 / lam_init)`

where `lam_init = 2 / sigma2` unless you explicitly pass `--lam`.

## Example 1: default-style paper grid

```bash
PYTHONPATH=/mnt/data python /mnt/data/run_l1_tpca_grid_v2.py \
  --shape 6,6,6 --ranks 2,2,2 \
  --N 60 --sigma2 1.0 \
  --replications 3 \
  --vbem_iter 200 --power_iter 50 --gaussian_iter 40 \
  --device cpu \
  --output_prefix paper_grid
```

## Example 2: smaller quick grid

```bash
PYTHONPATH=/mnt/data python /mnt/data/run_l1_tpca_grid_v2.py \
  --scenarios tensor_outliers,hybrid \
  --update_lam_values 0,1 \
  --z_prior_precisions 1.0,10.0,50.0 \
  --prior_schemes none,mild \
  --replications 2 \
  --device cpu \
  --output_prefix quick_grid
```

## Example 3: explore more hyperparameters

```bash
PYTHONPATH=/mnt/data python /mnt/data/run_l1_tpca_grid_v2.py \
  --weight_damping_values 0.0,0.3 \
  --mstep_inner_values 1,5 \
  --run_two_stage_values 0,1 \
  --two_stage_blend_values 0.5,1.0 \
  --replications 2 \
  --device cpu \
  --output_prefix extended_grid
```

## CSV structure

### Raw CSV

Each row is one run for one seed and one hyperparameter configuration.
The row includes:

- configuration columns such as `config.scenario`, `config.update_lam`, `config.z_prior_precision`, etc.
- all main metrics from `robust_vbem`, `robust_power`, `gaussian_em`, and (if enabled) `two_stage_vbem`
- detection summaries such as top-k precision/recall and outlier-to-clean ratios
- convergence flags and iteration counts

### Summary CSV

Each row aggregates all replications for one hyperparameter configuration.
It reports means / medians / minima / maxima for key metrics such as:

- factor recovery errors
- top-k detection accuracy for weight and delta
- outlier-to-clean score ratios
- iteration counts and convergence flags

## Notes

- The script catches run-time errors and writes them into the raw CSV with `status=error`.
- Use `--fail_fast` if you want the grid to stop immediately on the first error.
- The default grid can take a long time. Reduce the grid or replications for quick tests.
