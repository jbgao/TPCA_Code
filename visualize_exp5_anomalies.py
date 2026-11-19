from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def frame_files(folder: Path) -> list[Path]:
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

    def key(path: Path) -> tuple[int, str]:
        try:
            return int(path.stem), path.name
        except ValueError:
            return 10**9, path.name

    return sorted((p for p in folder.iterdir() if p.suffix.lower() in exts), key=key)


def load_gray(path: Path, size: tuple[int, int]) -> Image.Image:
    return Image.open(path).convert("L").resize(size)


def fonts() -> tuple[ImageFont.ImageFont, ImageFont.ImageFont, ImageFont.ImageFont]:
    try:
        return (
            ImageFont.truetype("arial.ttf", 30),
            ImageFont.truetype("arial.ttf", 22),
            ImageFont.truetype("arial.ttf", 18),
        )
    except OSError:
        f = ImageFont.load_default()
        return f, f, f


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 5
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=(255, 255, 255))
    draw.text((x, y), text, fill=(20, 20, 20), font=font)


def contact_sheet(rows: list[dict[str, Any]], test_root: Path, outpath: Path, *, n: int, resize: tuple[int, int]) -> None:
    title_font, font, small = fonts()
    tile_w, tile_h = resize
    cols = 4
    use_rows = rows[:n]
    cell_w, cell_h = tile_w + 38, tile_h + 78
    header_h = 70
    sheet = Image.new("RGB", (cols * cell_w + 40, header_h + ((len(use_rows) + cols - 1) // cols) * cell_h + 25), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((25, 18), "UCSD Ped2: top L1-TPCA anomaly-window representative frames", fill=(20, 20, 20), font=title_font)

    for k, row in enumerate(use_rows):
        clip = row["clip"]
        start = int(row["start_frame"])
        rank = int(row["rank"])
        files = frame_files(test_root / clip)
        frame_idx = min(start + 5, len(files) - 1)
        img = load_gray(files[frame_idx], resize).convert("RGB")

        c = k % cols
        r = k // cols
        x = 20 + c * cell_w
        y = header_h + r * cell_h
        sheet.paste(img, (x + 18, y + 18))
        draw.rectangle((x + 18, y + 18, x + 18 + tile_w, y + 18 + tile_h), outline=(40, 40, 40), width=2)
        draw_label(draw, (x + 26, y + 26), f"#{rank}", font)
        draw.text((x + 18, y + tile_h + 26), f"{clip}, frame {frame_idx + 1}", fill=(20, 20, 20), font=small)
        draw.text((x + 18, y + tile_h + 49), f"L1 score {row['score_rmse']:.4f}", fill=(60, 60, 60), font=small)

    sheet.save(outpath)


def frame_strips(rows: list[dict[str, Any]], test_root: Path, outdir: Path, *, n: int, resize: tuple[int, int]) -> None:
    title_font, font, small = fonts()
    tile_w, tile_h = resize
    offsets = [0, 2, 5, 7, 9]
    for row in rows[:n]:
        clip = row["clip"]
        start = int(row["start_frame"])
        rank = int(row["rank"])
        files = frame_files(test_root / clip)
        width = len(offsets) * (tile_w + 18) + 26
        height = tile_h + 118
        strip = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(strip)
        draw.text(
            (18, 14),
            f"Rank {rank}: {clip}, window start frame {start + 1}, L1 score {row['score_rmse']:.4f}",
            fill=(20, 20, 20),
            font=title_font,
        )
        for j, off in enumerate(offsets):
            idx = min(start + off, len(files) - 1)
            img = load_gray(files[idx], resize).convert("RGB")
            x = 18 + j * (tile_w + 18)
            y = 68
            strip.paste(img, (x, y))
            draw.rectangle((x, y, x + tile_w, y + tile_h), outline=(40, 40, 40), width=2)
            draw.text((x, y + tile_h + 8), f"frame {idx + 1}", fill=(40, 40, 40), font=small)
        strip.save(outdir / f"exp5_anomaly_rank{rank:02d}_{clip}_start{start:03d}.png")


def main() -> None:
    ap = argparse.ArgumentParser(description="Create visual diagnostics for top Exp. 5 video anomaly windows.")
    ap.add_argument("--results", type=str, default="results/exp5_video_anomaly_realdata.json")
    ap.add_argument("--test_root", type=str, default="data/video/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test")
    ap.add_argument("--outdir", type=str, default="figures")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--strips", type=int, default=4)
    ap.add_argument("--resize", type=str, default="180,180")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    w, h = (int(x) for x in args.resize.split(","))
    payload = json.loads(Path(args.results).read_text(encoding="utf-8"))
    rows = payload["top_l1_test_windows"]
    test_root = Path(args.test_root)
    contact_sheet(rows, test_root, outdir / "exp5_top_anomaly_frames_contact_sheet.png", n=args.top, resize=(w, h))
    frame_strips(rows, test_root, outdir, n=args.strips, resize=(w, h))


if __name__ == "__main__":
    main()
