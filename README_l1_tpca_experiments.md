# L1-TPCA experiment programs

This folder now has five experiment drivers for the L1-TPCA paper.

## Experiment 1: correctly specified model

```powershell
py -3.11 exp1_model_specification.py --replications 3 --max_iter 30
```

Vary sample size and Laplace scale-mixture precision, then compare L1-TPCA with a Gaussian TPCA baseline by subspace error and reconstruction RMSE.

## Experiment 2: contamination robustness

```powershell
py -3.11 exp2_contamination_robustness.py --replications 3 --max_iter 30
```

Inject whole-tensor and slice outliers. The output includes subspace recovery, RMSE, and L1 posterior-weight diagnostics for clean versus contaminated samples.

## Experiment 3: computation and scalability

```powershell
py -3.11 exp3_computation_scalability.py --max_iter 15
```

Compare runtime and convergence for the exact posterior-second-moment M-step, the ECM approximation, and Gaussian TPCA.

## Experiment 4: hyperspectral real data

Download a small public hyperspectral dataset:

```powershell
py -3.11 download_l1_tpca_data.py --dataset indian_pines
```

Then run:

```powershell
py -3.11 exp4_hyperspectral_realdata.py --dataset indian_pines --max_iter 20
```

This experiment uses hyperspectral image patches as tensors and reports reconstruction plus nearest-centroid classification from latent-core features. It needs `scipy.io.loadmat` to read `.mat` files.

## Experiment 5: video anomaly real data

The UCSD Ped dataset is large. Download:

```powershell
py -3.11 download_l1_tpca_data.py --dataset ucsd_ped
```

Extract the archive, then point the script at the normal training and test frame-folder roots, for example:

```powershell
py -3.11 exp5_video_anomaly_realdata.py `
  --train_root data/video/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train `
  --test_root data/video/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test
```

The output lists the highest-scoring anomalous test windows by reconstruction residual.

## Convert JSON outputs to paper figures

After the full experiments finish, run:

```powershell
py -3.11 plot_l1_tpca_results.py `
  --exp1 results/exp1_model_specification.json `
  --exp2 results/exp2_contamination_robustness.json `
  --exp3 results/exp3_computation_scalability.json `
  --exp4 results/exp4_hyperspectral_indian_pines.json results/exp4_hyperspectral_salinas.json `
  --exp5 results/exp5_video_anomaly_realdata.json `
  --skip_missing `
  --outdir figures
```

This writes CSV summaries and, when Matplotlib is available, PNG figures:

- `exp1_subspace_error_vs_N.png`
- `exp1_rmse_vs_N.png`
- `exp2_error_vs_contamination_rate.png`
- `exp2_weight_boxplot.png`
- `exp2_delta_boxplot.png`
- `exp3_runtime_vs_N.png`
- `exp3_runtime_vs_dimension.png`
- `exp4_<dataset>_bars.png`
- `exp5_top_video_windows.png`

If Matplotlib is not available, export summaries only:

```powershell
py -3.11 plot_l1_tpca_results.py --skip_missing --csv_only --outdir figures
```

For a hyperspectral reconstruction panel, run in a Matplotlib-enabled environment:

```powershell
py -3.11 plot_hyperspectral_reconstruction.py --dataset indian_pines --output figures/indian_pines_reconstruction_panel.png
py -3.11 plot_hyperspectral_reconstruction.py --dataset salinas --output figures/salinas_reconstruction_panel.png
```
