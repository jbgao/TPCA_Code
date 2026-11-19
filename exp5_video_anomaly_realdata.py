from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from exp_l1_tpca_common import parse_shape, write_json
from l1_tpca_numpy import GaussianTPCAALS, L1TPCAVBEM, latent_posterior_means, reconstruction_errors


def read_gray(path: Path, size: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("L").resize(size)
    return np.asarray(img, dtype=float) / 255.0


def frame_files(folder: Path):
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


def load_clip_cubes(folder: Path, *, resize: tuple[int, int], window: int, stride: int, max_cubes: int | None):
    files = frame_files(folder)
    frames = [read_gray(p, resize) for p in files]
    cubes = []
    starts = []
    for s in range(0, max(0, len(frames) - window + 1), stride):
        cube = np.stack(frames[s : s + window], axis=2)
        cubes.append(cube)
        starts.append(s)
        if max_cubes is not None and len(cubes) >= max_cubes:
            break
    return cubes, starts


def load_many_clips(
    root: Path,
    *,
    resize: tuple[int, int],
    window: int,
    stride: int,
    max_cubes_per_clip: int | None,
    include_ground_truth_masks: bool = False,
):
    samples = []
    clip_names = []
    starts = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        if not include_ground_truth_masks and folder.name.lower().endswith("_gt"):
            continue
        cubes, ss = load_clip_cubes(folder, resize=resize, window=window, stride=stride, max_cubes=max_cubes_per_clip)
        samples.extend(cubes)
        clip_names.extend([folder.name] * len(cubes))
        starts.extend(ss)
    return samples, clip_names, starts


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Experiment 5: video anomaly tensors, e.g. UCSD Ped1/Ped2 frame folders.")
    ap.add_argument("--train_root", type=str, required=True, help="Folder containing normal training clip subfolders.")
    ap.add_argument("--test_root", type=str, required=True, help="Folder containing test clip subfolders.")
    ap.add_argument("--resize", type=parse_shape, default="48,48", help="width,height")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max_train_cubes_per_clip", type=int, default=20)
    ap.add_argument("--max_test_cubes_per_clip", type=int, default=30)
    ap.add_argument("--ranks", type=parse_shape, default="4,4,2")
    ap.add_argument("--max_iter", type=int, default=30)
    ap.add_argument("--mstep_inner", type=int, default=1)
    ap.add_argument("--seed", type=int, default=500)
    ap.add_argument("--output", type=str, default="results/exp5_video_anomaly_realdata.json")
    args = ap.parse_args()

    resize = (int(args.resize[0]), int(args.resize[1]))
    train_samples, train_clips, train_starts = load_many_clips(
        Path(args.train_root),
        resize=resize,
        window=args.window,
        stride=args.stride,
        max_cubes_per_clip=args.max_train_cubes_per_clip,
    )
    test_samples, test_clips, test_starts = load_many_clips(
        Path(args.test_root),
        resize=resize,
        window=args.window,
        stride=args.stride,
        max_cubes_per_clip=args.max_test_cubes_per_clip,
    )
    if not train_samples or not test_samples:
        raise RuntimeError("No frame cubes found. Point --train_root and --test_root to folders with clip subfolders of images.")

    l1 = L1TPCAVBEM(args.ranks, max_iter=args.max_iter, mstep_inner=args.mstep_inner, seed=args.seed).fit(train_samples)
    gauss = GaussianTPCAALS(args.ranks, max_iter=args.max_iter, mstep_inner=args.mstep_inner, seed=args.seed).fit(train_samples)

    l1_test_z = latent_posterior_means(test_samples, l1.mean, l1.factors)
    g_test_z = latent_posterior_means(test_samples, gauss.mean, gauss.factors, noise_variance=gauss.sigma2)
    l1_resid = reconstruction_errors(test_samples, l1.mean, l1.factors, l1_test_z)
    g_resid = reconstruction_errors(test_samples, gauss.mean, gauss.factors, g_test_z)

    # Score test windows by residual; low VBEM weights are available for training diagnostics,
    # while test residuals provide a model-agnostic frame-window anomaly score.
    top = np.argsort(-l1_resid)[: min(20, len(l1_resid))]
    payload = {
        "config": vars(args),
        "n_train": len(train_samples),
        "n_test": len(test_samples),
        "l1_tpca": {
            "rho_hat": float(l1.rho),
            "train_rmse": float(np.mean(reconstruction_errors(train_samples, l1.mean, l1.factors, l1.z_means))),
            "test_rmse_mean": float(l1_resid.mean()),
            "test_rmse_q95": float(np.quantile(l1_resid, 0.95)),
            "lowest_training_weight_indices": np.argsort(l1.weights)[: min(20, len(l1.weights))].astype(int).tolist(),
        },
        "gaussian_tpca": {
            "sigma2_hat": float(gauss.sigma2),
            "train_rmse": float(np.mean(reconstruction_errors(train_samples, gauss.mean, gauss.factors, gauss.z_means))),
            "test_rmse_mean": float(g_resid.mean()),
            "test_rmse_q95": float(np.quantile(g_resid, 0.95)),
        },
        "top_l1_test_windows": [
            {
                "rank": int(r + 1),
                "index": int(i),
                "clip": test_clips[i],
                "start_frame": int(test_starts[i]),
                "score_rmse": float(l1_resid[i]),
                "gaussian_rmse": float(g_resid[i]),
            }
            for r, i in enumerate(top)
        ],
    }
    write_json(payload, args.output)
