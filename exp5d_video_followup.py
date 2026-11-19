r"""Experiment 5 follow-up (pre-registered 2026-11-17): UCSD Ped2 with contaminated training.

Same data, splits, contamination and fits as 2026-11-17-corrected-reruns / noise-floor exp5c
(5 random Test clips supply abnormal training windows; the other 7 are evaluated), with the
noise floor (kappa 1e-3).  Three analyses from the same fits:

  D1 training-set cleaning : AUROC for identifying the contaminated training windows.
     Scores: L1-TPCA 1 / w_i (posterior weight); Gaussian TPCA EM training residual RMSE;
     HOSVD projection residual; distance to the training mean.
  D2 subspace robustness   : mode-wise subspace distance between the fit on contaminated
     training (eps > 0) and the fit on clean training (eps = 0) of the same split, per method.
  D3 local scoring         : window score = max over 8x8-pixel spatial cells (6x6 = 36 cells,
     all 10 frames) of the cell residual RMSE; frame score = max over covering windows;
     frame-level AUROC / AP; same rule for every method.

Heartbeat every TPCA_HEARTBEAT_SECS seconds.  Run with python -u.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import l1_tpca_numpy as T
from rerun_corrected import (_l1_test_posterior, auroc, frame_eval, start_heartbeat, status,
                             window_frames_abnormal)

T.GRAM_ORDER = "c"
KAPPA, RANKS, WINDOW, STRIDE, CELL = 1e-3, (4, 4, 2), 10, 5, 8


def cell_score(R: np.ndarray) -> float:
    h, w, _ = R.shape
    best = 0.0
    for i in range(0, h, CELL):
        for j in range(0, w, CELL):
            c = R[i:i + CELL, j:j + CELL, :]
            best = max(best, float(np.sqrt(np.mean(c * c))))
    return best


def run(outdir: Path, data_dir: Path) -> None:
    from PIL import Image
    from exp5_video_anomaly_realdata import load_clip_cubes, load_many_clips

    base = data_dir / "video" / "UCSD_Anomaly_Dataset.v1p2" / "UCSDped2"
    status("loading Ped2")
    train_normal, _, _ = load_many_clips(base / "Train", resize=(48, 48), window=WINDOW, stride=STRIDE, max_cubes_per_clip=20)
    test_clips = sorted(p.name for p in (base / "Test").iterdir() if p.is_dir() and not p.name.endswith("_gt"))
    windows, labels = {}, {}
    for c in test_clips:
        cubes, starts = load_clip_cubes(base / "Test" / c, resize=(48, 48), window=WINDOW, stride=STRIDE, max_cubes=None)
        gt = sorted(f for f in (base / "Test" / f"{c}_gt").iterdir() if f.suffix.lower() in {".bmp", ".png", ".tif"})
        labels[c] = np.array([np.asarray(Image.open(f).convert("L")).max() > 0 for f in gt], dtype=bool)
        windows[c] = (cubes, starts)
    rows = []
    for split in (0, 1, 2):
        rng = np.random.default_rng(1000 + split)  # identical split/contamination draws to exp5c
        source = sorted(rng.choice(test_clips, size=5, replace=False).tolist())
        evalc = [c for c in test_clips if c not in source]
        pool = [X for c in source for X, s in zip(*windows[c]) if window_frames_abnormal(labels[c], s, WINDOW) >= 5]
        ev_X, ev_c, ev_s = [], [], []
        for c in evalc:
            for X, s in zip(*windows[c]):
                ev_X.append(X); ev_c.append(c); ev_s.append(s)
        clean_fits = {}
        for eps in (0.0, 0.1, 0.2, 0.3):
            n_add = min(int(round(eps / (1 - eps) * len(train_normal))), len(pool))
            idx = rng.choice(len(pool), size=n_add, replace=False) if n_add else np.array([], dtype=int)
            train = train_normal + [pool[i] for i in idx]
            is_c = np.r_[np.zeros(len(train_normal), bool), np.ones(n_add, bool)]
            seed = 500 + split
            status(f"split={split} eps={eps}: fitting on {len(train)} windows")
            l1 = T.L1TPCAVBEM(RANKS, max_iter=100, mstep_inner=1, seed=seed, normalize_mode="balance",
                              second_moment_method="kron", tau0_kappa=KAPPA).fit(train)
            g = T.GaussianTPCAEM(RANKS, max_iter=100, mstep_inner=1, seed=seed).fit(train)
            mu_h = np.mean(np.stack(train), axis=0)
            U = T.hosvd_warm_start([X - mu_h for X in train], RANKS, scale_to_sqrt_n=False)
            fits = {"l1": l1.factors, "gauss_em": g.factors, "hosvd": U}
            if eps == 0.0:
                clean_fits = fits
            status(f"split={split} eps={eps}: scoring")
            # residual tensors on evaluation windows
            l1_z, _ = _l1_test_posterior(ev_X, l1.mean, l1.factors, l1.rho, 30, l1.tau0)
            g_z = T.latent_posterior_means(ev_X, g.mean, g.factors, noise_variance=g.sigma2)
            R = {
                "l1": [X - l1.mean - T.multi_mode_dot(z, l1.factors) for X, z in zip(ev_X, l1_z)],
                "gauss_em": [X - g.mean - T.multi_mode_dot(z, g.factors) for X, z in zip(ev_X, g_z)],
                "hosvd": [(X - mu_h) - T.multi_mode_dot(T.multi_mode_dot(X - mu_h, U, transpose=True), U) for X in ev_X],
                "mean_only": [X - mu_h for X in ev_X],
            }
            row = {"split": split, "eps_target": eps, "eps_actual": n_add / len(train), "n_train": len(train), "n_contam": n_add,
                   "source_clips": source, "eval_clips": evalc}
            # D3 local scoring and whole-window reference
            row["D3_local"] = {m: frame_eval([cell_score(r) for r in Rs], ev_c, ev_s, WINDOW, labels) for m, Rs in R.items()}
            row["whole_window"] = {m: frame_eval([float(np.sqrt(np.mean(r * r))) for r in Rs], ev_c, ev_s, WINDOW, labels) for m, Rs in R.items()}
            # D1 training-set cleaning
            if n_add:
                g_tr_z = T.latent_posterior_means(train, g.mean, g.factors, noise_variance=g.sigma2)
                scores = {
                    "l1_inverse_weight": 1.0 / l1.weights,
                    "gauss_em_residual": T.reconstruction_errors(train, g.mean, g.factors, g_tr_z),
                    "hosvd_residual": np.array([np.linalg.norm((X - mu_h) - T.multi_mode_dot(T.multi_mode_dot(X - mu_h, U, transpose=True), U)) for X in train]),
                    "distance_to_mean": np.array([np.linalg.norm(X - mu_h) for X in train]),
                }
                row["D1_cleaning_auroc"] = {k: auroc(v, is_c) for k, v in scores.items()}
                # D2 subspace shift relative to the clean fit
                row["D2_subspace_shift"] = {m: T.overall_subspace_error(fits[m], clean_fits[m])["Err"] for m in fits}
            row["l1_rho"] = float(l1.rho); row["l1_tau0"] = float(l1.tau0)
            rows.append(row)
            print(json.dumps({"split": split, "eps": eps,
                              "D1": {k: round(v, 3) for k, v in row.get("D1_cleaning_auroc", {}).items()},
                              "D2": {k: round(v, 4) for k, v in row.get("D2_subspace_shift", {}).items()},
                              "D3": {m: round(v["auroc"], 3) for m, v in row["D3_local"].items()}}), flush=True)
            (outdir / "exp5d_followup.json").write_text(json.dumps({"rows": rows}, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--outdir", default="../experiments/2026-11-17-exp5-followup/results")
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    start_heartbeat("exp5d")
    t0 = time.time()
    run(out, Path(args.data_dir))
    status(f"exp5d done in {time.time() - t0:.0f}s")
