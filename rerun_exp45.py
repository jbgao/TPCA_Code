r"""Rerun of Experiments 4 (hyperspectral) and 5 (UCSD Ped2 video), 2026-11-17.

Two modes, same data, splits, seeds, ranks and iteration budgets as the submission:
  --mode legacy  : the ORIGINAL implementation (frozen copy l1_tpca_numpy_original.py)
                   -> should reproduce results_original/*.json on this machine;
  --mode revised : the corrected implementation (l1_tpca_numpy.py):
                   quadrature GIG moments (no Bessel overflow), correct Kronecker Gram
                   ordering, exact Kronecker second-moment M-step for all sizes,
                   'balance' factor normalisation (L unchanged), and the exact,
                   monotone Gaussian TPCA EM baseline (GaussianTPCAEM).

Extra diagnostics (both modes): fitted rho, iterations/convergence, spread of the
training weights, the PPCA bound ||x - xhat|| <= ||x - mu|| on test samples, and
for Exp 4 model-based test weights for clean vs corrupted patches (revised only)
and RMSE against the clean test patches; for Exp 5 frame-level AUROC / AP against
the UCSD Ped2 ground-truth masks (frame score = max over covering windows).

Long-running: prints a heartbeat every TPCA_HEARTBEAT_SECS seconds (default 60; 0 = off).
Run with python -u.  Example:
  python -u rerun_exp45.py --exp 4 --dataset indian_pines --mode revised
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import threading
import time
from pathlib import Path

import numpy as np

_STATUS = {"now": "starting"}


def status(msg: str) -> None:
    _STATUS["now"] = msg
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def start_heartbeat(name: str) -> None:
    secs = float(os.environ.get("TPCA_HEARTBEAT_SECS", "60"))
    if secs <= 0:
        return
    t0 = time.time()

    def run() -> None:
        while True:
            time.sleep(secs)
            print(f"[heartbeat {time.strftime('%H:%M:%S')}] {name} alive, elapsed {time.time() - t0:.0f}s, now: {_STATUS['now']}", flush=True)

    threading.Thread(target=run, daemon=True).start()


# ---------------------------------------------------------------- models
def load_impl(mode: str):
    if mode == "legacy":
        return importlib.import_module("l1_tpca_numpy_original")
    T = importlib.import_module("l1_tpca_numpy")
    T.GRAM_ORDER = "c"
    return T


def make_models(T, mode: str, ranks, max_iter: int, mstep_inner: int, seed: int):
    if mode == "legacy":
        l1 = T.L1TPCAVBEM(ranks, max_iter=max_iter, mstep_inner=mstep_inner, seed=seed)
        g = T.GaussianTPCAALS(ranks, max_iter=max_iter, mstep_inner=mstep_inner, seed=seed)
    else:
        l1 = T.L1TPCAVBEM(ranks, max_iter=max_iter, mstep_inner=mstep_inner, seed=seed,
                          normalize_mode="balance", second_moment_method="kron", verbose=True)
        g = T.GaussianTPCAEM(ranks, max_iter=max_iter, mstep_inner=mstep_inner, seed=seed, verbose=True)
    return l1, g


def l1_model_based_test(T, samples, mean, factors, rho, iters: int = 30):
    """Per-sample fixed point of the variational E-step at fitted (mu, A, rho)."""
    ranks = tuple(A.shape[1] for A in factors)
    q = int(np.prod(ranks))
    p = int(np.prod(samples[0].shape))
    nu = 1.0 - 0.5 * p
    G = T.gram_kron(factors)
    w = np.ones(len(samples))
    rhs = [T.multi_mode_dot(X - mean, factors, transpose=True).reshape(-1) for X in samples]
    for _ in range(iters):
        zs, deltas = [], []
        for X, r, wi in zip(samples, rhs, w):
            S = np.linalg.solve(np.eye(q) + wi * G, np.eye(q))
            z = (wi * (S @ r)).reshape(ranks)
            R = X - mean - T.multi_mode_dot(z, factors)
            zs.append(z)
            deltas.append(float(np.sum(R * R)) + float(np.trace(G @ S)))
        _, w_new, _ = T.gig_moments(nu, rho, np.asarray(deltas))
        if np.max(np.abs(w_new - w) / np.maximum(w, 1e-300)) < 1e-10:
            w = w_new
            break
        w = w_new
    return zs, w


def bound_fraction(T, samples, mean, factors, zs) -> float:
    ok = [np.linalg.norm(X - mean - T.multi_mode_dot(z, factors)) <= np.linalg.norm(X - mean) * (1 + 1e-9) for X, z in zip(samples, zs)]
    return float(np.mean(ok))


def wstats(w) -> dict:
    w = np.asarray(w, dtype=float)
    return {"min": float(w.min()), "median": float(np.median(w)), "max": float(w.max()),
            "max_over_min": float(w.max() / max(w.min(), 1e-300))}


def fit_info(model, res, kind: str) -> dict:
    info = {"n_iter": int(res.n_iter), "converged": bool(res.converged),
            "objective_first": float(res.objective_path[0]) if res.objective_path else None,
            "objective_last": float(res.objective_path[-1]) if res.objective_path else None}
    if kind == "gauss" and hasattr(model, "monotone_violation"):
        info["monotone_violation"] = bool(model.monotone_violation)
    return info


# ---------------------------------------------------------------- Exp 4
def run_exp4(args, T) -> dict:
    from exp4_hyperspectral_realdata import corrupt_test_patches, extract_patches, load_mat_array, standardize_cube
    from exp_l1_tpca_common import classification_accuracy

    root = Path(args.data_dir) / "hyperspectral"
    if args.dataset == "indian_pines":
        cube = load_mat_array(root / "Indian_pines_corrected.mat", ("indian_pines_corrected", "indian_pines"))
        gt = load_mat_array(root / "Indian_pines_gt.mat", ("indian_pines_gt",))
        stride = 9
    else:
        cube = load_mat_array(root / "Salinas_corrected.mat", ("salinas_corrected", "salinas"))
        gt = load_mat_array(root / "Salinas_gt.mat", ("salinas_gt",))
        stride = 24  # as in results_original/exp4_hyperspectral_salinas.json
    ranks = (3, 3, 4)
    seed, max_iter, mstep_inner = 400, 30, 1
    status(f"loading {args.dataset}")
    cube = standardize_cube(cube)
    samples, labels = extract_patches(cube, gt, patch_size=9, stride=stride, max_patches=300, seed=seed)
    rng = np.random.default_rng(seed)
    n_train = int(round(0.7 * len(samples)))
    train, test = samples[:n_train], samples[n_train:]
    train_y, test_y = labels[:n_train], labels[n_train:]
    test_c, mask = corrupt_test_patches(test, rng, rate=0.1, scale=8.0)

    l1, g = make_models(T, args.mode, ranks, max_iter, mstep_inner, seed)
    status(f"fitting L1-TPCA ({args.mode}) on {len(train)} patches of shape {train[0].shape}")
    l1r = l1.fit(train)
    status(f"fitting Gaussian TPCA ({args.mode})")
    gr = g.fit(train)
    status("evaluating")

    # original test protocol: L1 with w = 1, Gaussian with its fitted noise variance
    l1_test_z = T.latent_posterior_means(test_c, l1r.mean, l1r.factors)
    g_test_z = T.latent_posterior_means(test_c, gr.mean, gr.factors, noise_variance=gr.sigma2)
    feat = lambda zs: np.stack([z.reshape(-1) for z in zs], axis=0)
    out = {
        "config": {"exp": 4, "dataset": args.dataset, "mode": args.mode, "ranks": ranks, "patch_size": 9, "stride": stride,
                   "max_patches": 300, "train_fraction": 0.7, "test_corruption_rate": 0.1, "test_corruption_scale": 8.0,
                   "max_iter": max_iter, "mstep_inner": mstep_inner, "seed": seed},
        "n_train": len(train), "n_test": len(test), "test_corrupt_count": int(mask.sum()),
        "l1_tpca": {
            "rho_hat": float(l1r.rho),
            "train_rmse": float(np.mean(T.reconstruction_errors(train, l1r.mean, l1r.factors, l1r.z_means))),
            "test_rmse": float(np.mean(T.reconstruction_errors(test_c, l1r.mean, l1r.factors, l1_test_z))),
            "test_rmse_vs_clean": float(np.mean(T.reconstruction_errors(test, l1r.mean, l1r.factors, l1_test_z))),
            "nearest_centroid_accuracy": classification_accuracy(feat(l1r.z_means), train_y, feat(l1_test_z), test_y),
            "train_weights": wstats(l1r.weights),
            "test_bound_fraction": bound_fraction(T, test_c, l1r.mean, l1r.factors, l1_test_z),
            "fit": fit_info(l1, l1r, "l1"),
        },
        "gaussian_tpca": {
            "sigma2_hat": float(gr.sigma2),
            "train_rmse": float(np.mean(T.reconstruction_errors(train, gr.mean, gr.factors, gr.z_means))),
            "test_rmse": float(np.mean(T.reconstruction_errors(test_c, gr.mean, gr.factors, g_test_z))),
            "test_rmse_vs_clean": float(np.mean(T.reconstruction_errors(test, gr.mean, gr.factors, g_test_z))),
            "nearest_centroid_accuracy": classification_accuracy(feat(gr.z_means), train_y, feat(g_test_z), test_y),
            "test_bound_fraction": bound_fraction(T, test_c, gr.mean, gr.factors, g_test_z),
            "fit": fit_info(g, gr, "gauss"),
        },
    }
    if args.mode == "revised":
        zs, w = l1_model_based_test(T, test_c, l1r.mean, l1r.factors, l1r.rho)
        out["l1_tpca"]["model_based_test"] = {
            "test_rmse": float(np.mean(T.reconstruction_errors(test_c, l1r.mean, l1r.factors, zs))),
            "test_rmse_vs_clean": float(np.mean(T.reconstruction_errors(test, l1r.mean, l1r.factors, zs))),
            "weight_clean_mean": float(np.mean(w[~mask])), "weight_corrupt_mean": float(np.mean(w[mask])),
            "weights": wstats(w),
        }
    return out


# ---------------------------------------------------------------- Exp 5
def frame_labels(gt_dir: Path):
    from PIL import Image
    files = sorted(p for p in gt_dir.iterdir() if p.suffix.lower() in {".bmp", ".png", ".tif", ".tiff"})
    return np.array([np.asarray(Image.open(f).convert("L")).max() > 0 for f in files], dtype=bool)


def auroc(scores, labels) -> float:
    from scipy.stats import rankdata
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    n1, n0 = labels.sum(), (~labels).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(scores, labels) -> float:
    order = np.argsort(-np.asarray(scores, float), kind="mergesort")
    y = np.asarray(labels, bool)[order]
    if y.sum() == 0:
        return float("nan")
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float(np.sum(prec[y]) / y.sum())


def frame_scores(window_scores, clips, starts, window: int):
    per = {}
    for s, c, st in zip(window_scores, clips, starts):
        d = per.setdefault(c, {})
        for f in range(st, st + window):
            d[f] = max(d.get(f, -np.inf), float(s))
    return per


def frame_eval(window_scores, clips, starts, window, test_root: Path) -> dict:
    per = frame_scores(window_scores, clips, starts, window)
    sc, lab = [], []
    for c, d in sorted(per.items()):
        labels = frame_labels(test_root / f"{c}_gt")
        for f, s in d.items():
            if f < len(labels):
                sc.append(s)
                lab.append(labels[f])
    lab = np.asarray(lab, bool)
    return {"frames": int(len(lab)), "abnormal_frames": int(lab.sum()), "auroc": auroc(sc, lab), "ap": average_precision(sc, lab),
            "ap_chance": float(lab.mean())}


def run_exp5(args, T) -> dict:
    from exp5_video_anomaly_realdata import load_many_clips

    base = Path(args.data_dir) / "video" / "UCSD_Anomaly_Dataset.v1p2" / "UCSDped2"
    ranks, seed, max_iter, mstep_inner, window, stride = (4, 4, 2), 500, 30, 1, 10, 5
    status("loading UCSD Ped2 windows")
    train, _, _ = load_many_clips(base / "Train", resize=(48, 48), window=window, stride=stride, max_cubes_per_clip=20)
    test, clips, starts = load_many_clips(base / "Test", resize=(48, 48), window=window, stride=stride, max_cubes_per_clip=30)
    l1, g = make_models(T, args.mode, ranks, max_iter, mstep_inner, seed)
    status(f"fitting L1-TPCA ({args.mode}) on {len(train)} windows")
    l1r = l1.fit(train)
    status(f"fitting Gaussian TPCA ({args.mode})")
    gr = g.fit(train)
    status("evaluating")
    l1_z = T.latent_posterior_means(test, l1r.mean, l1r.factors)
    g_z = T.latent_posterior_means(test, gr.mean, gr.factors, noise_variance=gr.sigma2)
    l1_res = T.reconstruction_errors(test, l1r.mean, l1r.factors, l1_z)
    g_res = T.reconstruction_errors(test, gr.mean, gr.factors, g_z)
    # mean-only reference detector: distance to the training mean frame-window
    mean_res = np.array([np.linalg.norm(X - np.mean(np.stack(train), axis=0)) / math.sqrt(X.size) for X in test])
    out = {
        "config": {"exp": 5, "mode": args.mode, "ranks": ranks, "resize": [48, 48], "window": window, "stride": stride,
                   "max_train_cubes_per_clip": 20, "max_test_cubes_per_clip": 30, "max_iter": max_iter,
                   "mstep_inner": mstep_inner, "seed": seed},
        "n_train": len(train), "n_test": len(test),
        "l1_tpca": {
            "rho_hat": float(l1r.rho),
            "train_rmse": float(np.mean(T.reconstruction_errors(train, l1r.mean, l1r.factors, l1r.z_means))),
            "test_rmse_mean": float(l1_res.mean()), "test_rmse_q95": float(np.quantile(l1_res, 0.95)),
            "train_weights": wstats(l1r.weights),
            "test_bound_fraction": bound_fraction(T, test, l1r.mean, l1r.factors, l1_z),
            "frame_level": frame_eval(l1_res, clips, starts, window, base / "Test"),
            "fit": fit_info(l1, l1r, "l1"),
        },
        "gaussian_tpca": {
            "sigma2_hat": float(gr.sigma2),
            "train_rmse": float(np.mean(T.reconstruction_errors(train, gr.mean, gr.factors, gr.z_means))),
            "test_rmse_mean": float(g_res.mean()), "test_rmse_q95": float(np.quantile(g_res, 0.95)),
            "test_bound_fraction": bound_fraction(T, test, gr.mean, gr.factors, g_z),
            "frame_level": frame_eval(g_res, clips, starts, window, base / "Test"),
            "fit": fit_info(g, gr, "gauss"),
        },
        "mean_only_reference": {"frame_level": frame_eval(mean_res, clips, starts, window, base / "Test"),
                                "test_rmse_mean": float(mean_res.mean())},
        "top10_l1_windows": [{"clip": clips[i], "start_frame": int(starts[i]), "score": float(l1_res[i])} for i in np.argsort(-l1_res)[:10]],
    }
    if args.mode == "revised":
        zs, w = l1_model_based_test(T, test, l1r.mean, l1r.factors, l1r.rho)
        mres = T.reconstruction_errors(test, l1r.mean, l1r.factors, zs)
        out["l1_tpca"]["model_based_test"] = {"test_rmse_mean": float(mres.mean()), "weights": wstats(w),
                                              "frame_level_residual": frame_eval(mres, clips, starts, window, base / "Test"),
                                              "frame_level_inverse_weight": frame_eval(1.0 / w, clips, starts, window, base / "Test")}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exp", type=int, choices=[4, 5], required=True)
    ap.add_argument("--dataset", choices=["indian_pines", "salinas"], default="indian_pines")
    ap.add_argument("--mode", choices=["legacy", "revised"], required=True)
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--outdir", type=str, default="../experiments/2026-11-17-rerun-exp45/results")
    args = ap.parse_args()
    tag = f"exp4_{args.dataset}" if args.exp == 4 else "exp5_ucsd_ped2"
    start_heartbeat(f"rerun_exp45 {tag} {args.mode}")
    T = load_impl(args.mode)
    t0 = time.time()
    payload = run_exp4(args, T) if args.exp == 4 else run_exp5(args, T)
    payload["seconds"] = time.time() - t0
    out = Path(args.outdir) / f"{tag}_{args.mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    status(f"wrote {out} ({payload['seconds']:.0f}s)")
