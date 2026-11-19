from __future__ import annotations

import argparse
from pathlib import Path

import requests


DATASETS = {
    "indian_pines": {
        "root": "data/hyperspectral",
        "files": {
            "Indian_pines_corrected.mat": "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat",
            "Indian_pines_gt.mat": "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat",
        },
    },
    "salinas": {
        "root": "data/hyperspectral",
        "files": {
            "Salinas_corrected.mat": "http://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat",
            "Salinas_gt.mat": "http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat",
        },
    },
    "ucsd_ped": {
        "root": "data/video",
        "files": {
            "UCSD_Anomaly_Dataset.tar.gz": "http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz",
        },
        "note": "Large download, about 707 MB. Extract it manually or with tar before running exp5.",
    },
}


def download(url: str, dest: Path, *, chunk_size: int = 1024 * 1024, force: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"exists: {dest}")
        return
    part = dest.with_suffix(dest.suffix + ".part")
    if part.exists():
        part.unlink()
    print(f"downloading {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", "0") or 0)
        got = 0
        with part.open("wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                got += len(chunk)
                if total:
                    print(f"  {dest.name}: {100 * got / total:5.1f}%", end="\r")
    if total and got != total:
        raise RuntimeError(f"incomplete download for {dest.name}: got {got} of {total} bytes")
    part.replace(dest)
    print(f"saved: {dest}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download public datasets used by the L1-TPCA experiments.")
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="indian_pines")
    ap.add_argument("--data_root", type=str, default=None, help="Override target root folder.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing files after a successful fresh download.")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    root = Path(args.data_root or cfg["root"])
    if "note" in cfg:
        print(cfg["note"])
    for name, url in cfg["files"].items():
        download(url, root / name, force=args.force)

