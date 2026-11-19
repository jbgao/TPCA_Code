from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from exp4_hyperspectral_realdata import extract_patches, load_mat_array, standardize_cube


def font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


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


def scale_rgb(cube: np.ndarray, bands: tuple[int, int, int]) -> Image.Image:
    chans = []
    for b in bands:
        chans.append(np.asarray(scale_gray(cube[:, :, b], robust=True), dtype=np.uint8))
    return Image.fromarray(np.stack(chans, axis=2))


def add_title(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    draw.text(xy, text, fill=(25, 25, 25), font=font(24))


def main() -> None:
    ap = argparse.ArgumentParser(description="Create robust-scaled hyperspectral data diagnostics for Experiment 4.")
    ap.add_argument("--dataset", choices=["indian_pines", "salinas"], default="indian_pines")
    ap.add_argument("--data_dir", type=str, default="data/hyperspectral")
    ap.add_argument("--outdir", type=str, default="figures")
    args = ap.parse_args()

    root = Path(args.data_dir)
    out = Path(args.outdir) / f"{args.dataset}_data_diagnostic.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.dataset == "indian_pines":
        cube_raw = load_mat_array(root / "Indian_pines_corrected.mat", ("indian_pines_corrected", "indian_pines"))
        gt = load_mat_array(root / "Indian_pines_gt.mat", ("indian_pines_gt",))
        display_name = "Indian Pines"
        rgb_bands = (30, 60, 90)
        single_band = 30
    else:
        cube_raw = load_mat_array(root / "Salinas_corrected.mat", ("salinas_corrected", "salinas"))
        gt = load_mat_array(root / "Salinas_gt.mat", ("salinas_gt",))
        display_name = "Salinas"
        rgb_bands = (30, 90, 160)
        single_band = 90
    cube = standardize_cube(cube_raw)

    samples, _ = extract_patches(cube, gt, patch_size=17, stride=17, max_patches=120, seed=900)
    patch_scores = np.asarray([x.std() for x in samples])
    patch_idx = int(np.argmax(patch_scores))
    patch = samples[patch_idx]
    rng = np.random.default_rng(900)
    band = single_band
    corrupted = patch.copy()
    corrupted[:, :, band] += rng.normal(scale=8.0, size=corrupted[:, :, band].shape)

    full_rgb = scale_rgb(cube, rgb_bands).resize((360, 360), Image.Resampling.NEAREST)
    gt_img = scale_gray(gt, robust=False).convert("RGB").resize((360, 360), Image.Resampling.NEAREST)
    band_img = scale_gray(cube[:, :, band], robust=True).convert("RGB").resize((360, 360), Image.Resampling.NEAREST)
    patch_img = scale_gray(patch[:, :, band], robust=True).convert("RGB").resize((260, 260), Image.Resampling.NEAREST)
    corrupt_img = scale_gray(corrupted[:, :, band], robust=True).convert("RGB").resize((260, 260), Image.Resampling.NEAREST)
    diff_img = scale_gray(np.abs(corrupted[:, :, band] - patch[:, :, band]), robust=True).convert("RGB").resize((260, 260), Image.Resampling.NEAREST)

    canvas = Image.new("RGB", (1180, 850), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 24), f"{display_name} data diagnostic: robust-scaled views", fill=(15, 15, 15), font=font(34))
    draw.text((30, 66), "False-color, single-band, label, patch, and corruption views.", fill=(70, 70, 70), font=font(20))

    panels = [
        (full_rgb, (30, 120), f"False color bands {rgb_bands[0]}/{rgb_bands[1]}/{rgb_bands[2]}"),
        (band_img, (410, 120), f"Single band {band}, full image"),
        (gt_img, (790, 120), "Ground-truth labels"),
        (patch_img, (105, 530), f"Patch {patch_idx}, band {band}"),
        (corrupt_img, (460, 530), "Same patch after corruption"),
        (diff_img, (815, 530), "|corruption| map"),
    ]
    for img, xy, title in panels:
        canvas.paste(img, xy)
        x, y = xy
        draw.rectangle((x, y, x + img.width, y + img.height), outline=(35, 35, 35), width=2)
        add_title(draw, (x, y - 34), title)

    draw.text(
        (30, 810),
        f"Raw cube shape={cube_raw.shape}; raw range=({cube_raw.min():.0f}, {cube_raw.max():.0f}); selected patch std={patch.std():.3f}.",
        fill=(55, 55, 55),
        font=font(20),
    )
    canvas.save(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
