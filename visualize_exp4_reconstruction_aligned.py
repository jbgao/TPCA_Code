from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from exp4_hyperspectral_realdata import extract_patches, load_mat_array, standardize_cube
from l1_tpca_numpy import GaussianTPCAALS, L1TPCAVBEM, latent_posterior_means, multi_mode_dot


def font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def reconstruct_one(mean: np.ndarray, factors: list[np.ndarray], z: np.ndarray) -> np.ndarray:
    return mean + multi_mode_dot(z, factors)


def scale_gray(x: np.ndarray, *, robust: bool = True) -> Image.Image:
    x = np.asarray(x, dtype=float)
    if robust:
        lo, hi = np.percentile(x, [2, 98])
    else:
        lo, hi = float(np.min(x)), float(np.max(x))
    if hi <= lo:
        hi = lo + 1e-8
    y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return Image.fromarray(np.uint8(np.round(255 * y)))


def draw_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    img: np.ndarray,
    xy: tuple[int, int],
    title: str,
    subtitle: str,
    *,
    size: int = 235,
) -> None:
    x, y = xy
    rendered = scale_gray(img, robust=True).convert("RGB").resize((size, size), Image.Resampling.NEAREST)
    canvas.paste(rendered, (x, y))
    draw.rectangle((x, y, x + size, y + size), outline=(35, 35, 35), width=2)
    draw.text((x, y - 54), title, fill=(20, 20, 20), font=font(24))
    if subtitle:
        draw.text((x, y - 27), subtitle, fill=(65, 65, 65), font=font(17))


def load_dataset(dataset: str, root: Path) -> tuple[np.ndarray, np.ndarray, str, int]:
    if dataset == "indian_pines":
        cube = load_mat_array(root / "Indian_pines_corrected.mat", ("indian_pines_corrected", "indian_pines"))
        gt = load_mat_array(root / "Indian_pines_gt.mat", ("indian_pines_gt",))
        return cube, gt, "Indian Pines", 30
    cube = load_mat_array(root / "Salinas_corrected.mat", ("salinas_corrected", "salinas"))
    gt = load_mat_array(root / "Salinas_gt.mat", ("salinas_gt",))
    return cube, gt, "Salinas", 90


def main() -> None:
    ap = argparse.ArgumentParser(description="Create aligned robust-scaled Exp. 4 reconstruction panels.")
    ap.add_argument("--dataset", choices=["indian_pines", "salinas"], default="salinas")
    ap.add_argument("--data_dir", type=str, default="data/hyperspectral")
    ap.add_argument("--outdir", type=str, default="figures")
    ap.add_argument("--patch_size", type=int, default=17)
    ap.add_argument("--stride", type=int, default=17)
    ap.add_argument("--max_patches", type=int, default=120)
    ap.add_argument("--train_fraction", type=float, default=0.7)
    ap.add_argument("--ranks", type=str, default="3,3,4")
    ap.add_argument("--max_iter", type=int, default=30)
    ap.add_argument("--mstep_inner", type=int, default=1)
    ap.add_argument("--seed", type=int, default=900)
    args = ap.parse_args()

    ranks = tuple(int(x) for x in args.ranks.split(","))
    cube_raw, gt, display_name, band = load_dataset(args.dataset, Path(args.data_dir))
    cube = standardize_cube(cube_raw)
    samples, _ = extract_patches(
        cube,
        gt,
        patch_size=args.patch_size,
        stride=args.stride,
        max_patches=args.max_patches,
        seed=args.seed,
    )
    patch_scores = np.asarray([x.std() for x in samples])
    selected_global_idx = int(np.argmax(patch_scores))
    n_train = int(round(args.train_fraction * len(samples)))
    if selected_global_idx < n_train:
        selected_global_idx = int(n_train + np.argmax(patch_scores[n_train:]))
    selected_test_idx = selected_global_idx - n_train

    train = samples[:n_train]
    clean = samples[selected_global_idx].copy()
    rng = np.random.default_rng(args.seed)
    corrupted = clean.copy()
    corrupted[:, :, band] += rng.normal(scale=8.0, size=corrupted[:, :, band].shape)

    l1 = L1TPCAVBEM(ranks, max_iter=args.max_iter, mstep_inner=args.mstep_inner, seed=args.seed).fit(train)
    gauss = GaussianTPCAALS(ranks, max_iter=args.max_iter, mstep_inner=args.mstep_inner, seed=args.seed).fit(train)
    z_l1 = latent_posterior_means([corrupted], l1.mean, l1.factors)[0]
    z_g = latent_posterior_means([corrupted], gauss.mean, gauss.factors, noise_variance=gauss.sigma2)[0]
    rec_l1 = reconstruct_one(l1.mean, l1.factors, z_l1)
    rec_g = reconstruct_one(gauss.mean, gauss.factors, z_g)

    err_g = np.abs(clean[:, :, band] - rec_g[:, :, band])
    err_l1 = np.abs(clean[:, :, band] - rec_l1[:, :, band])
    full_err_g = float(np.sqrt(np.mean((clean - rec_g) ** 2)))
    full_err_l1 = float(np.sqrt(np.mean((clean - rec_l1) ** 2)))
    band_err_g = float(np.sqrt(np.mean(err_g**2)))
    band_err_l1 = float(np.sqrt(np.mean(err_l1**2)))

    width, height = 1180, 780
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 24), f"{display_name}: aligned reconstruction panel", fill=(15, 15, 15), font=font(34))
    draw.text(
        (28, 66),
        f"Selected patch {selected_global_idx} (test index {selected_test_idx}), band {band}; all panels use robust per-panel display scaling.",
        fill=(65, 65, 65),
        font=font(20),
    )

    xs = [55, 335, 615, 895]
    y1, y2 = 160, 500
    draw_panel(canvas, draw, clean[:, :, band], (xs[0], y1), "Original", "standardized band patch")
    draw_panel(canvas, draw, corrupted[:, :, band], (xs[1], y1), "Corrupted", "same band after perturbation")
    draw_panel(canvas, draw, rec_g[:, :, band], (xs[2], y1), "Gaussian TPCA", f"band RMSE {band_err_g:.3g}")
    draw_panel(canvas, draw, rec_l1[:, :, band], (xs[3], y1), "L1-TPCA", f"band RMSE {band_err_l1:.3g}")
    draw_panel(canvas, draw, np.abs(corrupted[:, :, band] - clean[:, :, band]), (xs[0], y2), "|Corruption|", "injected band noise")
    draw_panel(canvas, draw, err_g, (xs[1], y2), "|Gaussian error|", f"full RMSE {full_err_g:.3g}")
    draw_panel(canvas, draw, err_l1, (xs[2], y2), "|L1 error|", f"full RMSE {full_err_l1:.3g}")

    draw.text(
        (895, 540),
        f"rho={l1.rho:.3g}\nGaussian sigma2={gauss.sigma2:.3g}\n"
        f"Patch std={clean.std():.3g}\n"
        "Display note:\nvalues are not rescaled\nfor RMSE computation.",
        fill=(35, 35, 35),
        font=font(21),
        spacing=6,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{args.dataset}_reconstruction_panel_aligned.png"
    canvas.save(out)
    print(f"saved {out}")
    print(f"selected_global_idx={selected_global_idx}, test_idx={selected_test_idx}, band={band}")
    print(f"Gaussian RMSE: band={band_err_g:.6g}, full={full_err_g:.6g}")
    print(f"L1-TPCA RMSE: band={band_err_l1:.6g}, full={full_err_l1:.6g}")


if __name__ == "__main__":
    main()
