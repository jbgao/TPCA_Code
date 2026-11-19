# L1-TPCA contamination benchmark v2

This patch adds two things:

1. richer **VBEM diagnostics** beyond low weight alone:
   - `delta` score (high is suspicious)
   - posterior latent norm `z_norm_sq` (high may indicate the model is using an extreme latent core)
   - `latent_adjusted_score = z_norm_sq / weight`
   - `combined_score = delta * sqrt(z_norm_sq) / weight`

2. a **two-stage pipeline**:
   - run `L1TPCARobustPower` first
   - use its factors, mean, and weights to initialize `L1TPCAVBEM`

## Files

- `l1_tpca_torch_v2.py`: patched VBEM implementation with:
  - `z_prior_precision`
  - optional lambda priors / damping
  - optional `init_weights`, `init_factors`, `init_mean`
  - new samplewise diagnostics in the result object
- `l1_tpca_contamination_v2.py`: benchmark harness with the new summaries and the optional two-stage solver
- `run_l1_tpca_contamination_v2.py`: command-line runner

## Example

```bash
PYTHONPATH=/mnt/data python /mnt/data/run_l1_tpca_contamination_v2.py   --scenario tensor_outliers   --shape 6,6,6 --ranks 2,2,2 --N 60   --sigma2 1.0 --contamination_rate 0.1 --outlier_scale 8.0   --vbem_iter 200 --mstep_inner 5 --power_iter 50 --gaussian_iter 40   --update_lam 0 --z_prior_precision 5.0 --run_two_stage 1 --two_stage_blend 1.0   --device cpu
```

## Reading the output

For `robust_vbem` and `two_stage_vbem`, check:

- `weight_summary`: low scores mean stronger downweighting
- `delta_summary`: high scores mean larger residual proxy
- `znorm_summary`: high scores mean larger posterior latent norm
- `latent_adjusted_summary`: high scores often catch samples that are fit by unusual latent cores
- `combined_summary`: a stronger hybrid anomaly score

If VBEM recovers the factors well but `topk_low_score_precision` stays poor, the latent-norm or combined summaries are often more informative than weight alone.
