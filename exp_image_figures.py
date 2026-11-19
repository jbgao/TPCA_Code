r"""Illustrative image arrays for Sections 6.2-6.3 (pre-registered 2026-11-17, see
experiments/2026-11-17-image-figures/config.json).

Hyperspectral: replicates exp4r_spatial.py fold 0, whole-patch N(0, 4^2) training contamination at
eps = 0.2, and reconstructs every 9x9 patch of the fold-0 test band; checks the refit against the
recorded rmse_vs_clean.  Video: replicates exp5d_video_followup.py split 0, clean training, and
computes local cell maps and frame scores for one evaluation clip chosen by a fixed rule; checks the
recorded D3 AUROC.  Arrays are written to <outdir>/*.npz for make_figures.py.

Heartbeat every TPCA_HEARTBEAT_SECS seconds (default 60).  Run with python -u.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import l1_tpca_numpy as T
from exp4_hyperspectral_realdata import load_mat_array
from exp4r_spatial import (BUFFER, KAPPA, MAX_TEST, MAX_TRAIN, PATCH, RANKS, STRIDE, TEST_RATE,
                           contaminate_train, corrupt_band, patches_in_columns, rmse)
from rerun_corrected import _l1_test_posterior, frame_eval, start_heartbeat, status, window_frames_abnormal

T.GRAM_ORDER = "c"
# Approximate true colour: both scenes are AVIRIS (about 10 nm spacing from about 400 nm; the removed
# water-absorption bands all lie above index 100), so indices 26/16/6 are near 650/550/450 nm.
BANDS = {"indian_pines": (26, 16, 6), "salinas": (26, 16, 6)}
EPS, KIND, SCALE, FOLD, SHOW_PATCH_ROWS = 0.2, "patch", 4.0, 0, 16


def robust_rgb(cube, bands, ref=None):
    """Colour composite with per-band 1/99 percentile scaling and gamma 0.9 (percentiles from ref, default the cube)."""
    ref = cube if ref is None else ref
    out = []
    for b in bands:
        lo, hi = np.percentile(ref[:, :, b], [1, 99])
        out.append(np.clip((cube[:, :, b] - lo) / max(hi - lo, 1e-12), 0, 1) ** 0.9)
    return np.stack(out, axis=2)


def hyperspectral(dataset: str, data_dir: Path, outdir: Path, heavy_dir: Path) -> dict:
    root = data_dir / "hyperspectral"
    if dataset == "indian_pines":
        cube = load_mat_array(root / "Indian_pines_corrected.mat", ("indian_pines_corrected", "indian_pines")).astype(float)
        gt = load_mat_array(root / "Indian_pines_gt.mat", ("indian_pines_gt",))
    else:
        cube = load_mat_array(root / "Salinas_corrected.mat", ("salinas_corrected", "salinas")).astype(float)
        gt = load_mat_array(root / "Salinas_gt.mat", ("salinas_gt",))
    H, W, _ = cube.shape
    fold = FOLD
    lo, hi = fold * W // 3, (fold + 1) * W // 3
    # ---- exact replica of exp4r_spatial.run for this fold and eps
    te_raw, te_y = patches_in_columns(cube, gt, lo, hi)
    tr_raw = []
    if lo - BUFFER > 0:
        tr_raw += patches_in_columns(cube, gt, 0, lo - BUFFER)[0]
    if hi + BUFFER < W:
        tr_raw += patches_in_columns(cube, gt, hi + BUFFER, W)[0]
    rng = np.random.default_rng(4000 + fold)
    tri = rng.permutation(len(tr_raw))[:MAX_TRAIN]
    tei = rng.permutation(len(te_raw))[:MAX_TEST]
    tr_raw = [tr_raw[i] for i in tri]
    te_sub = [te_raw[i] for i in tei]
    pix = np.concatenate([X.reshape(-1, X.shape[2]) for X in tr_raw])
    mu_b, sd_b = pix.mean(axis=0), np.maximum(pix.std(axis=0), 1e-8)
    train_clean = [(X - mu_b) / sd_b for X in tr_raw]
    test_clean = [(X - mu_b) / sd_b for X in te_sub]
    te_bad = rng.choice(len(test_clean), size=int(round(TEST_RATE * len(test_clean))), replace=False)
    test = corrupt_band(test_clean, te_bad, rng)
    rng_e = np.random.default_rng(5000 + 10 * fold + int(100 * EPS))
    n_bad = int(round(EPS * len(train_clean)))
    bad = rng_e.choice(len(train_clean), size=n_bad, replace=False)
    train = contaminate_train(train_clean, bad, rng_e, KIND, SCALE, 20)
    seed = 400 + fold
    status(f"{dataset}: L1 fit on {len(train)} patches ({n_bad} contaminated)")
    l1 = T.L1TPCAVBEM(RANKS, rho_init=1.0, max_iter=100, mstep_inner=1, seed=seed, normalize_mode="balance",
                      second_moment_method="kron", tau0_kappa=KAPPA).fit(train)
    status(f"{dataset}: Gaussian EM fit")
    g = T.GaussianTPCAEM(RANKS, max_iter=100, mstep_inner=1, seed=seed).fit(train)
    # ---- check against the recorded run
    l1_z, _ = _l1_test_posterior(test, l1.mean, l1.factors, l1.rho, 30, l1.tau0)
    g_z = T.latent_posterior_means(test, g.mean, g.factors, noise_variance=g.sigma2)
    chk = {"l1": float(np.mean([rmse(l1.mean + T.multi_mode_dot(z, l1.factors), c) for z, c in zip(l1_z, test_clean)])),
           "gauss_em": float(np.mean([rmse(g.mean + T.multi_mode_dot(z, g.factors), c) for z, c in zip(g_z, test_clean)]))}
    rec = [r for r in json.loads((heavy_dir / f"exp4r_{dataset}_patch_s4.json").read_text(encoding="utf-8"))["rows"]
           if r["fold"] == fold and abs(r["eps_train"] - EPS) < 1e-9][0]
    ref = {m: rec[m]["rmse_vs_clean"] for m in ("l1", "gauss_em")}
    ok = all(abs(chk[m] - ref[m]) <= 1e-6 * abs(ref[m]) for m in chk)
    print(json.dumps({"dataset": dataset, "refit_rmse_vs_clean": chk, "recorded": ref, "match": ok}), flush=True)
    # ---- reconstruct every patch of the test band (clean data), tile into an image
    nj = len(range(lo, hi - PATCH + 1, STRIDE))
    ni = min(len(range(0, H - PATCH + 1, STRIDE)), SHOW_PATCH_ROWS)
    status(f"{dataset}: reconstructing {ni * nj} region patches")
    region = cube[: ni * PATCH, lo: lo + nj * PATCH, :]
    patches = [(region[i * PATCH:(i + 1) * PATCH, j * PATCH:(j + 1) * PATCH, :] - mu_b) / sd_b for i in range(ni) for j in range(nj)]
    pl1, _ = _l1_test_posterior(patches, l1.mean, l1.factors, l1.rho, 30, l1.tau0)
    pg = T.latent_posterior_means(patches, g.mean, g.factors, noise_variance=g.sigma2)
    rec_l1 = np.zeros_like(region); rec_g = np.zeros_like(region); std_region = np.zeros_like(region)
    k = 0
    for i in range(ni):
        for j in range(nj):
            sl = (slice(i * PATCH, (i + 1) * PATCH), slice(j * PATCH, (j + 1) * PATCH))
            std_region[sl] = patches[k]
            rec_l1[sl] = l1.mean + T.multi_mode_dot(pl1[k], l1.factors)
            rec_g[sl] = g.mean + T.multi_mode_dot(pg[k], g.factors)
            k += 1
    err_l1 = np.sqrt(np.mean((rec_l1 - std_region) ** 2, axis=2))
    err_g = np.sqrt(np.mean((rec_g - std_region) ** 2, axis=2))
    bands = BANDS[dataset]
    out = outdir / f"hsi_{dataset}.npz"
    np.savez_compressed(
        out, rgb_full=robust_rgb(cube, bands), gt=gt.astype(np.int16), test_cols=np.array([lo, hi]), buffer=BUFFER,
        fold_edges=np.array([f * W // 3 for f in range(4)]), region_rows=ni * PATCH,
        rgb_clean=robust_rgb(region, bands, ref=cube),
        rgb_l1=robust_rgb(rec_l1 * sd_b + mu_b, bands, ref=cube),
        rgb_g=robust_rgb(rec_g * sd_b + mu_b, bands, ref=cube),
        err_l1=err_l1, err_g=err_g, bands=np.array(bands))
    return {"dataset": dataset, "npz": out.name, "refit": chk, "recorded": ref, "match": ok,
            "region_mean_rmse": {"l1": float(err_l1.mean()), "gauss_em": float(err_g.mean())},
            "region_rmse_patchwise": {"l1": float(np.mean([rmse(r, c) for r, c in zip([l1.mean + T.multi_mode_dot(z, l1.factors) for z in pl1], patches)])),
                                      "gauss_em": float(np.mean([rmse(r, c) for r, c in zip([g.mean + T.multi_mode_dot(z, g.factors) for z in pg], patches)]))},
            "region_shape": list(region.shape[:2])}


def cell_map(R: np.ndarray, cell: int = 8) -> np.ndarray:
    h, w, _ = R.shape
    return np.array([[np.sqrt(np.mean(R[i:i + cell, j:j + cell, :] ** 2)) for j in range(0, w, cell)] for i in range(0, h, cell)])


def video(data_dir: Path, outdir: Path, v5_dir: Path) -> dict:
    from PIL import Image
    from exp5_video_anomaly_realdata import frame_files, load_clip_cubes, load_many_clips

    WINDOW, VSTRIDE, VRANKS, split = 10, 5, (4, 4, 2), 0
    base = data_dir / "video" / "UCSD_Anomaly_Dataset.v1p2" / "UCSDped2"
    status("loading Ped2")
    train_normal, _, _ = load_many_clips(base / "Train", resize=(48, 48), window=WINDOW, stride=VSTRIDE, max_cubes_per_clip=20)
    test_clips = sorted(p.name for p in (base / "Test").iterdir() if p.is_dir() and not p.name.endswith("_gt"))
    windows, labels, gtfiles = {}, {}, {}
    for c in test_clips:
        windows[c] = load_clip_cubes(base / "Test" / c, resize=(48, 48), window=WINDOW, stride=VSTRIDE, max_cubes=None)
        gtfiles[c] = sorted(f for f in (base / "Test" / f"{c}_gt").iterdir() if f.suffix.lower() in {".bmp", ".png", ".tif"})
        labels[c] = np.array([np.asarray(Image.open(f).convert("L")).max() > 0 for f in gtfiles[c]], dtype=bool)
    rng = np.random.default_rng(1000 + split)
    source = sorted(rng.choice(test_clips, size=5, replace=False).tolist())
    evalc = [c for c in test_clips if c not in source]
    ev_X, ev_c, ev_s = [], [], []
    for c in evalc:
        for X, s in zip(*windows[c]):
            ev_X.append(X); ev_c.append(c); ev_s.append(s)
    seed = 500 + split
    status(f"video split {split}: L1 fit on {len(train_normal)} clean windows")
    l1 = T.L1TPCAVBEM(VRANKS, max_iter=100, mstep_inner=1, seed=seed, normalize_mode="balance",
                      second_moment_method="kron", tau0_kappa=KAPPA).fit(train_normal)
    status("video: Gaussian EM fit")
    g = T.GaussianTPCAEM(VRANKS, max_iter=100, mstep_inner=1, seed=seed).fit(train_normal)
    mu_h = np.mean(np.stack(train_normal), axis=0)
    status(f"video: residuals on {len(ev_X)} evaluation windows")
    l1_z, _ = _l1_test_posterior(ev_X, l1.mean, l1.factors, l1.rho, 30, l1.tau0)
    g_z = T.latent_posterior_means(ev_X, g.mean, g.factors, noise_variance=g.sigma2)
    R = {"l1": [X - l1.mean - T.multi_mode_dot(z, l1.factors) for X, z in zip(ev_X, l1_z)],
         "gauss_em": [X - g.mean - T.multi_mode_dot(z, g.factors) for X, z in zip(ev_X, g_z)],
         "mean_only": [X - mu_h for X in ev_X]}
    maps = {m: [cell_map(r) for r in Rs] for m, Rs in R.items()}
    wscore = {m: [float(cm.max()) for cm in ms_] for m, ms_ in maps.items()}
    chk = {m: frame_eval(wscore[m], ev_c, ev_s, WINDOW, labels)["auroc"] for m in wscore}
    rec = [r for r in json.loads((v5_dir / "exp5d_followup.json").read_text(encoding="utf-8"))["rows"]
           if r["split"] == split and r["eps_target"] == 0.0][0]
    ref = {m: rec["D3_local"][m]["auroc"] for m in chk}
    ok = all(abs(chk[m] - ref[m]) <= 1e-9 for m in chk)
    print(json.dumps({"video_refit_auroc": chk, "recorded": ref, "match": ok}), flush=True)
    # ---- fixed selection rules
    clip = max(evalc, key=lambda c: (int((~labels[c]).sum()), -evalc.index(c)))
    lab = labels[clip]
    areas = np.array([int((np.asarray(Image.open(f).convert("L")) > 0).sum()) for f in gtfiles[clip]])
    f_ab = int(np.argmax(areas))
    f_no = int(np.flatnonzero(~lab)[0])
    idx = [k for k, (c, s) in enumerate(zip(ev_c, ev_s)) if c == clip and s <= f_ab < s + WINDOW]
    k_win = min(idx, key=lambda k: (abs(ev_s[k] + (WINDOW - 1) / 2 - f_ab), ev_s[k]))
    files = frame_files(base / "Test" / clip)
    frame = lambda f: np.asarray(Image.open(files[f]).convert("L"), dtype=float) / 255.0
    # per-frame scores of this clip (max over covering windows), every method
    ks = [k for k, c in enumerate(ev_c) if c == clip]
    n_cov = max(ev_s[k] for k in ks) + WINDOW
    fs = {}
    for m in wscore:
        s = np.full(n_cov, -np.inf)
        for k in ks:
            s[ev_s[k]:ev_s[k] + WINDOW] = np.maximum(s[ev_s[k]:ev_s[k] + WINDOW], wscore[m][k])
        fs[m] = s
    out = outdir / "video_ped2.npz"
    np.savez_compressed(
        out, clip=clip, f_abnormal=f_ab, f_normal=f_no, win_start=ev_s[k_win],
        frame_normal=frame(f_no), frame_abnormal=frame(f_ab),
        gt_abnormal=(np.asarray(Image.open(gtfiles[clip][f_ab]).convert("L")) > 0),
        cell_l1=maps["l1"][k_win], cell_gauss=maps["gauss_em"][k_win], cell_mean=maps["mean_only"][k_win],
        score_l1=fs["l1"], score_gauss=fs["gauss_em"], score_mean=fs["mean_only"], labels=lab[:n_cov])
    return {"npz": out.name, "eval_clips": evalc, "clip": clip, "normal_frames_per_eval_clip": {c: int((~labels[c]).sum()) for c in evalc},
            "f_abnormal": f_ab, "f_normal": f_no, "win_start": int(ev_s[k_win]), "refit_auroc": chk, "recorded": ref, "match": ok}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--outdir", default="../experiments/2026-11-17-image-figures/results")
    ap.add_argument("--heavy_dir", default="../experiments/2026-11-17-exp4-heavier-contamination/results")
    ap.add_argument("--v5_dir", default="../experiments/2026-11-17-exp5-followup/results")
    ap.add_argument("--only", choices=["hsi", "video"], default=None)
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    start_heartbeat("exp_image_figures")
    t0 = time.time()
    summ_p = out / "image_figures_summary.json"
    summary = json.loads(summ_p.read_text(encoding="utf-8")) if summ_p.exists() else {}
    if args.only in (None, "hsi"):
        for ds in ("indian_pines", "salinas"):
            summary[ds] = hyperspectral(ds, Path(args.data_dir), out, Path(args.heavy_dir))
            summ_p.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    if args.only in (None, "video"):
        summary["video"] = video(Path(args.data_dir), out, Path(args.v5_dir))
        summ_p.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    status(f"exp_image_figures done in {time.time() - t0:.0f}s")
