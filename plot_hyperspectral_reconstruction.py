from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from exp4_hyperspectral_realdata import extract_patches, load_mat_array, standardize_cube
from exp_l1_tpca_common import parse_shape
from l1_tpca_numpy import GaussianTPCAALS, L1TPCAVBEM, latent_posterior_means, multi_mode_dot


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise RuntimeError("Matplotlib is required for this visualization script.") from exc
    return plt


def reconstruct_one(sample, mean, factors, z):
    return mean + multi_mode_dot(z, factors)


def corrupt_test_patches_with_bands(samples, rng, *, rate: float, scale: float):
    out = [X.copy() for X in samples]
    bands = np.full(len(out), -1, dtype=int)
    n_bad = int(round(rate * len(out)))
    mask = np.zeros(len(out), dtype=bool)
    if n_bad <= 0:
        return out, mask, bands
    for i in rng.choice(len(out), size=n_bad, replace=False):
        mask[i] = True
        band = int(rng.integers(0, out[i].shape[2]))
        bands[i] = band
        out[i][:, :, band] += rng.normal(scale=scale, size=out[i][:, :, band].shape)
    return out, mask, bands


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Make a four-panel hyperspectral reconstruction figure.")
    ap.add_argument("--dataset", choices=["indian_pines", "salinas"], default="indian_pines")
    ap.add_argument("--data_dir", type=str, default="data/hyperspectral")
    ap.add_argument("--ranks", type=parse_shape, default="3,3,4")
    ap.add_argument("--patch_size", type=int, default=17)
    ap.add_argument("--stride", type=int, default=17)
    ap.add_argument("--max_patches", type=int, default=120)
    ap.add_argument("--train_fraction", type=float, default=0.7)
    ap.add_argument("--test_corruption_rate", type=float, default=0.1)
    ap.add_argument("--test_corruption_scale", type=float, default=8.0)
    ap.add_argument("--band", type=str, default="auto")
    ap.add_argument("--select", choices=["largest_gap", "largest_gaussian_error", "last"], default="largest_gap")
    ap.add_argument("--max_iter", type=int, default=30)
    ap.add_argument("--mstep_inner", type=int, default=1)
    ap.add_argument("--seed", type=int, default=900)
    ap.add_argument("--output", type=str, default="figures/hyperspectral_reconstruction_panel.png")
    args = ap.parse_args()

    root = Path(args.data_dir)
    if args.dataset == "indian_pines":
        cube = load_mat_array(root / "Indian_pines_corrected.mat", ("indian_pines_corrected", "indian_pines"))
        gt = load_mat_array(root / "Indian_pines_gt.mat", ("indian_pines_gt",))
    else:
        cube = load_mat_array(root / "Salinas_corrected.mat", ("salinas_corrected", "salinas"))
        gt = load_mat_array(root / "Salinas_gt.mat", ("salinas_gt",))

    cube = standardize_cube(cube)
    samples, _ = extract_patches(
        cube,
        gt,
        patch_size=args.patch_size,
        stride=args.stride,
        max_patches=args.max_patches,
        seed=args.seed,
    )
    rng = np.random.default_rng(args.seed)
    n_train = int(round(args.train_fraction * len(samples)))
    train = samples[:n_train]
    test_clean = [X.copy() for X in samples[n_train:]]
    test_corrupted, corrupt_mask, corrupt_bands = corrupt_test_patches_with_bands(
        test_clean,
        rng,
        rate=args.test_corruption_rate,
        scale=args.test_corruption_scale,
    )

    l1 = L1TPCAVBEM(args.ranks, max_iter=args.max_iter, mstep_inner=args.mstep_inner, seed=args.seed).fit(train)
    gauss = GaussianTPCAALS(args.ranks, max_iter=args.max_iter, mstep_inner=args.mstep_inner, seed=args.seed).fit(train)
    z_l1_all = latent_posterior_means(test_corrupted, l1.mean, l1.factors)
    z_g_all = latent_posterior_means(test_corrupted, gauss.mean, gauss.factors, noise_variance=gauss.sigma2)
    rec_l1_all = [reconstruct_one(X, l1.mean, l1.factors, z) for X, z in zip(test_corrupted, z_l1_all)]
    rec_g_all = [reconstruct_one(X, gauss.mean, gauss.factors, z) for X, z in zip(test_corrupted, z_g_all)]
    rmse_l1_all = np.asarray([np.sqrt(np.mean((X - R) ** 2)) for X, R in zip(test_clean, rec_l1_all)])
    rmse_g_all = np.asarray([np.sqrt(np.mean((X - R) ** 2)) for X, R in zip(test_clean, rec_g_all)])
    candidate = np.flatnonzero(corrupt_mask)
    if candidate.size == 0:
        candidate = np.arange(len(test_clean))
    if args.select == "largest_gap":
        idx = int(candidate[np.argmax(rmse_g_all[candidate] - rmse_l1_all[candidate])])
    elif args.select == "largest_gaussian_error":
        idx = int(candidate[np.argmax(rmse_g_all[candidate])])
    else:
        idx = int(candidate[-1])

    clean = test_clean[idx]
    corrupted = test_corrupted[idx]
    rec_l1 = rec_l1_all[idx]
    rec_g = rec_g_all[idx]
    if args.band == "auto":
        if corrupt_bands[idx] >= 0:
            band = int(corrupt_bands[idx])
        else:
            band = int(np.argmax(np.mean((corrupted - clean) ** 2, axis=(0, 1))))
    else:
        band = min(int(args.band), clean.shape[2] - 1)
    err_g = np.abs(clean[:, :, band] - rec_g[:, :, band])
    err_l1 = np.abs(clean[:, :, band] - rec_l1[:, :, band])
    rmse_g = float(np.sqrt(np.mean(err_g ** 2)))
    rmse_l1 = float(np.sqrt(np.mean(err_l1 ** 2)))
    full_rmse_g = float(rmse_g_all[idx])
    full_rmse_l1 = float(rmse_l1_all[idx])

    panels = [
        ("Original", clean[:, :, band]),
        ("Corrupted", corrupted[:, :, band]),
        ("Gaussian TPCA", rec_g[:, :, band]),
        ("L1-TPCA", rec_l1[:, :, band]),
        (f"|Gaussian error|\nband={rmse_g:.3g}, full={full_rmse_g:.3g}", err_g),
        (f"|L1 error|\nband={rmse_l1:.3g}, full={full_rmse_l1:.3g}", err_l1),
    ]
    plt = require_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(8.2, 5.2))
    vmin = min(float(np.min(panels[i][1])) for i in range(4))
    vmax = max(float(np.max(panels[i][1])) for i in range(4))
    err_vmax = max(float(np.max(err_g)), float(np.max(err_l1)), 1e-12)
    for ax, (title, img) in zip(axes.ravel(), panels):
        if "error" in title:
            im = ax.imshow(img, cmap="magma", vmin=0.0, vmax=err_vmax)
        else:
            im = ax.imshow(img, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"{args.dataset} selected test patch {idx}, band {band}", y=0.99)
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")
    print(f"selected test patch={idx}, corrupted={bool(corrupt_mask[idx])}, band={band}")
    print(f"Gaussian RMSE: band={rmse_g:.6g}, full={full_rmse_g:.6g}")
    print(f"L1-TPCA RMSE: band={rmse_l1:.6g}, full={full_rmse_l1:.6g}")
