r"""Experiment 4 revised (R2 Major Comment 7), pre-registered 2026-11-17.

Hyperspectral patch experiment with
  * spatially separated splits: the image is cut into 3 vertical bands; each band in turn is
    the test region; training patches come from the other bands minus a 9-pixel buffer next
    to the test band; every patch lies entirely inside its region (no overlap, no leakage);
  * standardisation with training-patch statistics only;
  * training contamination eps_train in {0, 0.1, 0.2}: N(0, 8^2) added to one random band of
    that fraction of training patches (the corruption type of the original experiment);
    10% of test patches corrupted the same way (as in the original experiment);
  * methods: L1-TPCA with noise floor (kappa 1e-3), exact Gaussian TPCA EM, HOSVD projection;
    max_iter 100, mstep_inner 1, ranks (3,3,4);
  * reconstruction RMSE against the clean and against the corrupted test patches;
  * nearest-centroid classification with latent features (L1, Gaussian, HOSVD) and the
    requested baselines: majority class, centre-pixel spectrum, PCA features (same dimension);
    accuracy and balanced accuracy on labelled patches whose class occurs in training.

Heartbeat every TPCA_HEARTBEAT_SECS seconds (default 60).  Run with python -u.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import l1_tpca_numpy as T
from exp4_hyperspectral_realdata import load_mat_array
from rerun_corrected import _l1_test_posterior, start_heartbeat, status

T.GRAM_ORDER = "c"
PATCH, STRIDE, BUFFER, RANKS = 9, 9, 9, (3, 3, 4)
MAX_TRAIN, MAX_TEST, KAPPA, CORRUPT_SCALE, TEST_RATE = 300, 150, 1e-3, 8.0, 0.1


def patches_in_columns(cube, gt, col_lo, col_hi):
    h, w, _ = cube.shape
    out, labels = [], []
    for i in range(0, h - PATCH + 1, STRIDE):
        for j in range(col_lo, col_hi - PATCH + 1, STRIDE):
            out.append(cube[i:i + PATCH, j:j + PATCH, :].astype(float))
            lab = gt[i:i + PATCH, j:j + PATCH].reshape(-1)
            lab = lab[lab > 0]
            labels.append(int(np.bincount(lab.astype(int)).argmax()) if lab.size else 0)
    return out, np.asarray(labels, dtype=int)


def corrupt_band(samples, idx, rng):
    out = [X.copy() for X in samples]
    for i in idx:
        b = int(rng.integers(0, out[i].shape[2]))
        out[i][:, :, b] += rng.normal(scale=CORRUPT_SCALE, size=out[i][:, :, b].shape)
    return out


def contaminate_train(samples, idx, rng, kind, scale, nbands):
    """Training contamination (extension 2026-11-17):
    'band'      : N(0, 8^2) on one random band (original corruption type);
    'patch'     : N(0, scale^2) added to every entry of the patch (whole-observation);
    'multiband' : N(0, 8^2) on `nbands` random bands."""
    if kind == "band":
        return corrupt_band(samples, idx, rng)
    out = [X.copy() for X in samples]
    for i in idx:
        if kind == "patch":
            out[i] = out[i] + rng.normal(scale=scale, size=out[i].shape)
        elif kind == "multiband":
            bands = rng.choice(out[i].shape[2], size=nbands, replace=False)
            for b in bands:
                out[i][:, :, b] += rng.normal(scale=CORRUPT_SCALE, size=out[i][:, :, b].shape)
        else:
            raise ValueError(kind)
    return out


def rmse(A, B):
    return float(np.linalg.norm(A - B) / math.sqrt(A.size))


def nearest_centroid(train_f, train_y, test_f, test_y):
    keep_tr = train_y > 0
    classes = np.unique(train_y[keep_tr])
    keep_te = np.isin(test_y, classes)
    if keep_te.sum() == 0:
        return {"acc": float("nan"), "bal_acc": float("nan"), "n": 0}
    C = np.stack([train_f[keep_tr][train_y[keep_tr] == c].mean(axis=0) for c in classes])
    d = ((test_f[keep_te][:, None, :] - C[None]) ** 2).sum(axis=2)
    pred = classes[np.argmin(d, axis=1)]
    yt = test_y[keep_te]
    recalls = [float(np.mean(pred[yt == c] == c)) for c in np.unique(yt)]
    return {"acc": float(np.mean(pred == yt)), "bal_acc": float(np.mean(recalls)), "n": int(keep_te.sum())}


def run(dataset: str, data_dir: Path, outdir: Path, kind="band", scale=8.0, nbands=20, eps_list=(0.0, 0.1, 0.2), tag=""):
    root = data_dir / "hyperspectral"
    if dataset == "indian_pines":
        cube = load_mat_array(root / "Indian_pines_corrected.mat", ("indian_pines_corrected", "indian_pines")).astype(float)
        gt = load_mat_array(root / "Indian_pines_gt.mat", ("indian_pines_gt",))
    else:
        cube = load_mat_array(root / "Salinas_corrected.mat", ("salinas_corrected", "salinas")).astype(float)
        gt = load_mat_array(root / "Salinas_gt.mat", ("salinas_gt",))
    W = cube.shape[1]
    rows = []
    for fold in range(3):
        lo, hi = fold * W // 3, (fold + 1) * W // 3
        te_raw, te_y = patches_in_columns(cube, gt, lo, hi)
        tr_raw, tr_y = [], []
        if lo - BUFFER > 0:
            a, b = patches_in_columns(cube, gt, 0, lo - BUFFER); tr_raw += a; tr_y += list(b)
        if hi + BUFFER < W:
            a, b = patches_in_columns(cube, gt, hi + BUFFER, W); tr_raw += a; tr_y += list(b)
        tr_y = np.asarray(tr_y, dtype=int)
        rng = np.random.default_rng(4000 + fold)
        tri = rng.permutation(len(tr_raw))[:MAX_TRAIN]
        tei = rng.permutation(len(te_raw))[:MAX_TEST]
        tr_raw, tr_y = [tr_raw[i] for i in tri], tr_y[tri]
        te_raw, te_y = [te_raw[i] for i in tei], te_y[tei]
        # training-only bandwise standardisation
        pix = np.concatenate([X.reshape(-1, X.shape[2]) for X in tr_raw])
        mu_b, sd_b = pix.mean(axis=0), np.maximum(pix.std(axis=0), 1e-8)
        train_clean = [(X - mu_b) / sd_b for X in tr_raw]
        test_clean = [(X - mu_b) / sd_b for X in te_raw]
        n_te_bad = int(round(TEST_RATE * len(test_clean)))
        te_bad = rng.choice(len(test_clean), size=n_te_bad, replace=False)
        test = corrupt_band(test_clean, te_bad, rng)
        te_mask = np.zeros(len(test), bool); te_mask[te_bad] = True
        for eps in eps_list:
            rng_e = np.random.default_rng(5000 + 10 * fold + int(100 * eps))
            n_bad = int(round(eps * len(train_clean)))
            bad = rng_e.choice(len(train_clean), size=n_bad, replace=False) if n_bad else np.array([], int)
            train = contaminate_train(train_clean, bad, rng_e, kind, scale, nbands)
            tr_mask = np.zeros(len(train), bool); tr_mask[bad] = True
            seed = 400 + fold
            status(f"{dataset} fold={fold} eps={eps}: n_train={len(train)} n_test={len(test)}; L1 fit")
            l1 = T.L1TPCAVBEM(RANKS, rho_init=1.0, max_iter=100, mstep_inner=1, seed=seed, normalize_mode="balance",
                              second_moment_method="kron", tau0_kappa=KAPPA).fit(train)
            status(f"{dataset} fold={fold} eps={eps}: Gaussian EM fit")
            g = T.GaussianTPCAEM(RANKS, max_iter=100, mstep_inner=1, seed=seed).fit(train)
            status(f"{dataset} fold={fold} eps={eps}: evaluation")
            # reconstructions / features
            l1_te_z, l1_te_w = _l1_test_posterior(test, l1.mean, l1.factors, l1.rho, 30, l1.tau0)
            l1_tr_z, _ = _l1_test_posterior(train, l1.mean, l1.factors, l1.rho, 30, l1.tau0)
            g_te_z = T.latent_posterior_means(test, g.mean, g.factors, noise_variance=g.sigma2)
            g_tr_z = T.latent_posterior_means(train, g.mean, g.factors, noise_variance=g.sigma2)
            mu_h = np.mean(np.stack(train), axis=0)
            U = T.hosvd_warm_start([X - mu_h for X in train], RANKS, scale_to_sqrt_n=False)
            h_core = lambda X: T.multi_mode_dot(X - mu_h, U, transpose=True)
            h_rec = lambda X: mu_h + T.multi_mode_dot(h_core(X), U)

            def recon_stats(rec):
                r_c = [rmse(rec(i), test_clean[i]) for i in range(len(test))]
                r_x = [rmse(rec(i), test[i]) for i in range(len(test))]
                return {"rmse_vs_clean": float(np.mean(r_c)), "rmse_vs_input": float(np.mean(r_x)),
                        "rmse_vs_clean_corrupted_only": float(np.mean(np.asarray(r_c)[te_mask])) if te_mask.any() else None}

            res = {
                "dataset": dataset, "fold": fold, "train_contam": kind, "train_scale": scale, "nbands": nbands, "test_cols": [lo, hi], "eps_train": eps, "n_train": len(train),
                "n_test": len(test), "n_train_corrupt": int(n_bad), "n_test_corrupt": int(n_te_bad),
                "l1": {**recon_stats(lambda i: l1.mean + T.multi_mode_dot(l1_te_z[i], l1.factors)),
                       "rho_hat": float(l1.rho), "tau0": float(l1.tau0), "n_iter": int(l1.n_iter), "converged": bool(l1.converged),
                       "w_train_clean": float(l1.weights[~tr_mask].mean()),
                       "w_train_corrupt": float(l1.weights[tr_mask].mean()) if tr_mask.any() else None,
                       "w_test_clean": float(l1_te_w[~te_mask].mean()), "w_test_corrupt": float(l1_te_w[te_mask].mean()) if te_mask.any() else None},
                "gauss_em": {**recon_stats(lambda i: g.mean + T.multi_mode_dot(g_te_z[i], g.factors)),
                             "sigma2_hat": float(g.sigma2), "n_iter": int(g.n_iter), "converged": bool(g.converged)},
                "hosvd": recon_stats(lambda i: h_rec(test[i])),
            }
            # classification
            flat = lambda zs: np.stack([np.asarray(z).reshape(-1) for z in zs])
            Xtr = np.stack([X.reshape(-1) for X in train]); Xte = np.stack([X.reshape(-1) for X in test])
            mtr = Xtr.mean(axis=0)
            _, _, Vt = np.linalg.svd(Xtr - mtr, full_matrices=False)
            P = Vt[: int(np.prod(RANKS))].T
            feats = {
                "l1_latent": (flat(l1_tr_z), flat(l1_te_z)),
                "gauss_latent": (flat(g_tr_z), flat(g_te_z)),
                "hosvd_core": (flat([h_core(X) for X in train]), flat([h_core(X) for X in test])),
                "pca_features": ((Xtr - mtr) @ P, (Xte - mtr) @ P),
                "centre_pixel": (np.stack([X[PATCH // 2, PATCH // 2, :] for X in train]), np.stack([X[PATCH // 2, PATCH // 2, :] for X in test])),
            }
            cls = {k: nearest_centroid(a, tr_y, b, te_y) for k, (a, b) in feats.items()}
            lab_tr = tr_y[tr_y > 0]
            classes = np.unique(lab_tr)
            keep = np.isin(te_y, classes)
            maj = int(np.bincount(lab_tr).argmax()) if lab_tr.size else 0
            cls["majority_class"] = {"acc": float(np.mean(te_y[keep] == maj)) if keep.any() else float("nan"),
                                     "bal_acc": float(1.0 / len(np.unique(te_y[keep]))) if keep.any() else float("nan"), "n": int(keep.sum())}
            res["classification"] = cls
            res["n_test_labelled_in_train_classes"] = int(keep.sum())
            res["n_test_unlabelled"] = int((te_y == 0).sum())
            rows.append(res)
            print(json.dumps({"fold": fold, "eps": eps, "rmse_vs_clean": {m: round(res[m]["rmse_vs_clean"], 4) for m in ("l1", "gauss_em", "hosvd")},
                              "w_tr": [res["l1"]["w_train_clean"], res["l1"]["w_train_corrupt"]],
                              "acc": {k: round(v["acc"], 3) for k, v in cls.items()}}), flush=True)
            (outdir / f"exp4r_{dataset}{tag}.json").write_text(json.dumps({"rows": rows}, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["indian_pines", "salinas"], required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--outdir", default="../experiments/2026-11-17-exp4-revised/results")
    ap.add_argument("--train_contam", choices=["band", "patch", "multiband"], default="band")
    ap.add_argument("--train_scale", type=float, default=8.0)
    ap.add_argument("--nbands", type=int, default=20)
    ap.add_argument("--eps", default="0,0.1,0.2")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    start_heartbeat(f"exp4r {args.dataset} {args.train_contam}{args.tag}")
    t0 = time.time()
    run(args.dataset, Path(args.data_dir), out, kind=args.train_contam, scale=args.train_scale, nbands=args.nbands,
        eps_list=tuple(float(x) for x in args.eps.split(",")), tag=args.tag)
    status(f"exp4r {args.dataset} done in {time.time() - t0:.0f}s")
