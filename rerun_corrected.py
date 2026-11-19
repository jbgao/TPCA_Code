r"""Corrected reruns for the NEUCOM-D-26-22523 revision (pre-registered 2026-11-17).

Jobs (all with the corrected implementation l1_tpca_numpy.py, GRAM_ORDER = "c"):
  exp1  : Experiment 1 (correctly specified model), same data/seeds/protocol as the paper.
  exp2  : Experiment 2 (whole-tensor / slice contamination), same data/seeds/protocol.
          Arms for exp1/exp2:
            l1_orig  : L1-TPCA, original protocol (rho initialised at the truth, damped and
                       clipped to [rho/2, 2 rho], fixed-norm factors) but corrected numerics;
            l1_free  : L1-TPCA, rho_init = 1, no damping or bounds, 'balance' normalisation;
            gauss_em : exact monotone Gaussian TPCA EM ('balance' normalisation).
  exp5c : UCSD Ped2 with contaminated training.  Normal Train windows (20 per clip, as in the
          paper) plus a fraction eps of abnormal windows (>= 5 of 10 frames abnormal) drawn from
          5 randomly chosen Test clips; evaluation on all windows of the other 7 Test clips,
          frame-level AUROC / AP (frame score = max over covering windows).  Methods:
          L1-TPCA (revised; model-based test posterior), Gaussian TPCA EM, HOSVD projection,
          distance to the training mean.

Heartbeat every TPCA_HEARTBEAT_SECS seconds (default 60).  Run with python -u.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from pathlib import Path

import numpy as np

import l1_tpca_numpy as T
from exp2_contamination_robustness import contaminate_slices

T.GRAM_ORDER = "c"
_STATUS = {"now": "starting"}
TAG = ""
KAPPA = None  # noise-floor constant tau0_kappa; None = original model (no floor)


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


# ------------------------------------------------------------------ synthetic arms
def fit_arms(samples, true_factors, ranks, *, rho_true, max_iter, mstep_inner, seed):
    out = {}
    if KAPPA is not None:
        arms = {
            "l1_floor": T.L1TPCAVBEM(ranks, rho_init=1.0, max_iter=max_iter, mstep_inner=mstep_inner, seed=seed,
                                     exact_second_moment=True, normalize_mode="balance", second_moment_method="kron",
                                     tau0_kappa=KAPPA),
            "gauss_em": T.GaussianTPCAEM(ranks, max_iter=max_iter, mstep_inner=mstep_inner, seed=seed),
        }
        return _fit_arm_dict(arms, samples, true_factors, out)
    arms = {
        "l1_orig": T.L1TPCAVBEM(ranks, rho_init=rho_true, rho_min=0.5 * rho_true, rho_max=2.0 * rho_true,
                                rho_damping=0.5, max_iter=max_iter, mstep_inner=mstep_inner, seed=seed,
                                exact_second_moment=True, normalize_mode="fixed", second_moment_method="kron"),
        "l1_free": T.L1TPCAVBEM(ranks, rho_init=1.0, max_iter=max_iter, mstep_inner=mstep_inner, seed=seed,
                                exact_second_moment=True, normalize_mode="balance", second_moment_method="kron"),
        "gauss_em": T.GaussianTPCAEM(ranks, max_iter=max_iter, mstep_inner=mstep_inner, seed=seed),
    }
    return _fit_arm_dict(arms, samples, true_factors, out)


def _fit_arm_dict(arms, samples, true_factors, out):
    for name, model in arms.items():
        t0 = time.perf_counter()
        res = model.fit(samples)
        d = {
            "subspace_error": T.overall_subspace_error(res.factors, true_factors)["Err"],
            "rmse": float(np.mean(T.reconstruction_errors(samples, res.mean, res.factors, res.z_means))),
            "n_iter": int(res.n_iter), "converged": bool(res.converged), "seconds": time.perf_counter() - t0,
        }
        if name.startswith("l1"):
            d["rho_hat"] = float(res.rho)
            d["tau0"] = float(res.tau0)
            d["weights"] = res.weights.astype(float).tolist()
            d["w_max_over_median"] = float(res.weights.max() / np.median(res.weights))
            d["delta_min_over_median"] = float(res.delta.min() / np.median(res.delta))
        else:
            d["sigma2_hat"] = float(res.sigma2)
            d["monotone_violation"] = bool(model.monotone_violation)
        out[name] = d
    return out


def run_exp1(outdir: Path) -> None:
    shape, ranks, rows = (5, 5, 4), (2, 2, 2), []
    for N in (20, 40):
        for rho in (0.75, 2.0, 6.0):
            for rep in range(5):
                seed = 100 + 1000 * rep + 10 * N + int(100 * rho)  # exactly as exp1_model_specification.py
                rng = np.random.default_rng(seed)
                factors = T.random_factors(shape, ranks, rng)
                samples, _, _ = T.sample_l1_tpca(factors, rho, N, rng)
                status(f"exp1 N={N} rho={rho} rep={rep}")
                rows.append({"N": N, "rho_true": rho, "rep": rep,
                             **fit_arms(samples, factors, ranks, rho_true=rho, max_iter=150, mstep_inner=5, seed=seed)})
    (outdir / f"exp1_corrected{TAG}.json").write_text(json.dumps({"rows": rows}, indent=1) + "\n", encoding="utf-8")


def run_exp2(outdir: Path) -> None:
    shape, ranks, N, rho, rows = (5, 5, 4), (2, 2, 2), 50, 2.0, []
    for ctype in ("tensor", "slice"):
        for rate in (0.0, 0.05, 0.1, 0.2):
            for scale in (4.0, 8.0, 12.0):
                for rep in range(10):
                    seed = 200 + rep * 10000 + int(1000 * rate) + int(10 * scale)  # as exp2_contamination_robustness.py
                    rng = np.random.default_rng(seed)
                    factors = T.random_factors(shape, ranks, rng)
                    clean, _, _ = T.sample_l1_tpca(factors, rho, N, rng)
                    if ctype == "tensor":
                        samples, mask = T.contaminate_tensor_samples(clean, rng, contamination_rate=rate, outlier_scale=scale)
                    else:
                        samples, mask = contaminate_slices(clean, rng, contamination_rate=rate, outlier_scale=scale)
                    status(f"exp2 {ctype} rate={rate} scale={scale} rep={rep}")
                    arms = fit_arms(samples, factors, ranks, rho_true=rho, max_iter=40, mstep_inner=2, seed=seed)
                    for name in [k for k in arms if k.startswith("l1")]:
                        w = np.asarray(arms[name].pop("weights"))
                        arms[name]["weight_clean_mean"] = float(w[~mask].mean())
                        arms[name]["weight_outlier_mean"] = float(w[mask].mean()) if mask.any() else None
                    rows.append({"type": ctype, "rate": rate, "scale": scale, "rep": rep, "outliers": int(mask.sum()), **arms})
    (outdir / f"exp2_corrected{TAG}.json").write_text(json.dumps({"rows": rows}, indent=1) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ exp5c
def window_frames_abnormal(labels, start, window):
    return int(np.sum(labels[start:start + window]))


def auroc(scores, labels) -> float:
    from scipy.stats import rankdata
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    n1, n0 = labels.sum(), (~labels).sum()
    r = rankdata(scores)
    return float((r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(scores, labels) -> float:
    order = np.argsort(-np.asarray(scores, float), kind="mergesort")
    y = np.asarray(labels, bool)[order]
    prec = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float(np.sum(prec[y]) / y.sum())


def frame_eval(win_scores, clips, starts, window, labels_by_clip) -> dict:
    per = {}
    for s, c, st in zip(win_scores, clips, starts):
        d = per.setdefault(c, {})
        for f in range(st, st + window):
            d[f] = max(d.get(f, -np.inf), float(s))
    sc, lab = [], []
    for c, d in per.items():
        for f, s in d.items():
            if f < len(labels_by_clip[c]):
                sc.append(s)
                lab.append(labels_by_clip[c][f])
    lab = np.asarray(lab, bool)
    return {"frames": int(lab.size), "abnormal": int(lab.sum()), "auroc": auroc(sc, lab), "ap": average_precision(sc, lab),
            "ap_chance": float(lab.mean())}


def hosvd_projection_residual(train, test, ranks):
    mean = np.mean(np.stack(train), axis=0)
    U = T.hosvd_warm_start([X - mean for X in train], ranks, scale_to_sqrt_n=False)  # orthonormal bases
    res = []
    for X in test:
        C = X - mean
        P = T.multi_mode_dot(T.multi_mode_dot(C, U, transpose=True), U)
        res.append(float(np.linalg.norm(C - P) / math.sqrt(C.size)))
    return np.asarray(res)


def l1_model_based_residual(train_fit, test, iters=30):
    zs, _ = _l1_test_posterior(test, train_fit.mean, train_fit.factors, train_fit.rho, iters, train_fit.tau0)
    return T.reconstruction_errors(test, train_fit.mean, train_fit.factors, zs)


def _l1_test_posterior(samples, mean, factors, rho, iters, tau0=0.0):
    ranks = tuple(A.shape[1] for A in factors)
    q, p = int(np.prod(ranks)), int(np.prod(samples[0].shape))
    nu = 1.0 - 0.5 * p
    G = T.gram_kron(factors)
    rhs = [T.multi_mode_dot(X - mean, factors, transpose=True).reshape(-1) for X in samples]
    w = np.ones(len(samples))
    for _ in range(iters):
        zs, deltas = [], []
        for X, r, wi in zip(samples, rhs, w):
            S = np.linalg.solve(np.eye(q) + wi * G, np.eye(q))
            z = (wi * (S @ r)).reshape(ranks)
            R = X - mean - T.multi_mode_dot(z, factors)
            zs.append(z)
            deltas.append(float(np.sum(R * R)) + float(np.trace(G @ S)))
        _, w_new, _ = T.gig_moments(nu, rho, np.asarray(deltas), lower=tau0)
        done = np.max(np.abs(w_new - w) / np.maximum(w, 1e-300)) < 1e-10
        w = w_new
        if done:
            break
    return zs, w


def run_exp5c(outdir: Path, data_dir: Path) -> None:
    from PIL import Image
    from exp5_video_anomaly_realdata import load_clip_cubes, load_many_clips

    base = data_dir / "video" / "UCSD_Anomaly_Dataset.v1p2" / "UCSDped2"
    ranks, window, stride = (4, 4, 2), 10, 5
    status("loading Ped2 training windows")
    train_normal, _, _ = load_many_clips(base / "Train", resize=(48, 48), window=window, stride=stride, max_cubes_per_clip=20)
    test_clips = sorted(p.name for p in (base / "Test").iterdir() if p.is_dir() and not p.name.endswith("_gt"))
    windows, labels_by_clip = {}, {}
    for c in test_clips:
        cubes, starts = load_clip_cubes(base / "Test" / c, resize=(48, 48), window=window, stride=stride, max_cubes=None)
        gt = sorted(f for f in (base / "Test" / f"{c}_gt").iterdir() if f.suffix.lower() in {".bmp", ".png", ".tif"})
        labels_by_clip[c] = np.array([np.asarray(Image.open(f).convert("L")).max() > 0 for f in gt], dtype=bool)
        windows[c] = (cubes, starts)
    rows = []
    for split_seed in (0, 1, 2):
        rng = np.random.default_rng(1000 + split_seed)
        source = sorted(rng.choice(test_clips, size=5, replace=False).tolist())
        evalc = [c for c in test_clips if c not in source]
        pool = [X for c in source for X, s in zip(*windows[c]) if window_frames_abnormal(labels_by_clip[c], s, window) >= 5]
        ev_X, ev_c, ev_s = [], [], []
        for c in evalc:
            for X, s in zip(*windows[c]):
                ev_X.append(X); ev_c.append(c); ev_s.append(s)
        for eps in (0.0, 0.1, 0.2, 0.3):
            n_add = int(round(eps / (1 - eps) * len(train_normal)))
            n_add = min(n_add, len(pool))
            idx = rng.choice(len(pool), size=n_add, replace=False) if n_add else np.array([], dtype=int)
            train = train_normal + [pool[i] for i in idx]
            is_contam = np.r_[np.zeros(len(train_normal), bool), np.ones(n_add, bool)]
            eps_actual = n_add / len(train)
            seed = 500 + split_seed
            row = {"split_seed": split_seed, "source_clips": source, "eval_clips": evalc, "eps_target": eps,
                   "eps_actual": eps_actual, "n_train": len(train), "n_contam": n_add, "n_eval_windows": len(ev_X)}
            status(f"exp5c split={split_seed} eps={eps} (actual {eps_actual:.3f}) L1 fit on {len(train)} windows")
            l1 = T.L1TPCAVBEM(ranks, max_iter=100, mstep_inner=1, seed=seed, normalize_mode="balance",
                              second_moment_method="kron", tau0_kappa=KAPPA).fit(train)
            status(f"exp5c split={split_seed} eps={eps} Gaussian EM fit")
            g = T.GaussianTPCAEM(ranks, max_iter=100, mstep_inner=1, seed=seed).fit(train)
            status(f"exp5c split={split_seed} eps={eps} scoring {len(ev_X)} windows")
            s_l1 = l1_model_based_residual(l1, ev_X)
            gz = T.latent_posterior_means(ev_X, g.mean, g.factors, noise_variance=g.sigma2)
            s_g = T.reconstruction_errors(ev_X, g.mean, g.factors, gz)
            s_h = hosvd_projection_residual(train, ev_X, ranks)
            mu = np.mean(np.stack(train), axis=0)
            s_m = np.array([np.linalg.norm(X - mu) / math.sqrt(X.size) for X in ev_X])
            row["l1"] = {**frame_eval(s_l1, ev_c, ev_s, window, labels_by_clip), "rho_hat": float(l1.rho), "tau0": float(l1.tau0),
                         "n_iter": int(l1.n_iter), "converged": bool(l1.converged),
                         "w_normal_mean": float(l1.weights[~is_contam].mean()),
                         "w_contam_mean": float(l1.weights[is_contam].mean()) if n_add else None}
            row["gauss_em"] = {**frame_eval(s_g, ev_c, ev_s, window, labels_by_clip), "sigma2_hat": float(g.sigma2),
                               "n_iter": int(g.n_iter), "converged": bool(g.converged)}
            row["hosvd"] = frame_eval(s_h, ev_c, ev_s, window, labels_by_clip)
            row["mean_only"] = frame_eval(s_m, ev_c, ev_s, window, labels_by_clip)
            rows.append(row)
            print(json.dumps({k: (v["auroc"] if isinstance(v, dict) and "auroc" in v else v)
                              for k, v in row.items() if k in ("eps_actual", "l1", "gauss_em", "hosvd", "mean_only")}), flush=True)
            (outdir / f"exp5c_contaminated_training{TAG}.json").write_text(json.dumps({"rows": rows}, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", choices=["exp1", "exp2", "exp5c"], required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--outdir", default="../experiments/2026-11-17-corrected-reruns/results")
    ap.add_argument("--tau0_kappa", type=float, default=None, help="noise floor: tau0 = kappa * median per-entry variance")
    ap.add_argument("--tag", default="", help="suffix for output file names")
    args = ap.parse_args()
    KAPPA = args.tau0_kappa
    TAG = args.tag
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    start_heartbeat(f"rerun_corrected {args.job}")
    t0 = time.time()
    {"exp1": lambda: run_exp1(out), "exp2": lambda: run_exp2(out), "exp5c": lambda: run_exp5c(out, Path(args.data_dir))}[args.job]()
    status(f"{args.job} done in {time.time() - t0:.0f}s")
