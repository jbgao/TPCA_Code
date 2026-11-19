from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_path(obj: dict[str, Any], dotted: str, default: Any = np.nan) -> Any:
    cur: Any = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def as_float(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def grouped_summary(
    rows: list[dict[str, Any]],
    group_keys: list[str],
    value_getters: dict[str, Callable[[dict[str, Any]], float]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(row)
    out = []
    for key, vals in sorted(groups.items(), key=lambda kv: str(kv[0])):
        rec = {k: v for k, v in zip(group_keys, key)}
        rec["n"] = len(vals)
        for name, getter in value_getters.items():
            x = np.asarray([getter(v) for v in vals], dtype=float)
            x = x[np.isfinite(x)]
            rec[f"{name}_mean"] = float(np.mean(x)) if x.size else np.nan
            rec[f"{name}_se"] = float(np.std(x, ddof=1) / np.sqrt(x.size)) if x.size > 1 else 0.0
            rec[f"{name}_median"] = float(np.median(x)) if x.size else np.nan
        out.append(rec)
    return out


CSV_ONLY = False


def plot_exp5_pillow(payload: dict[str, Any], rows: list[dict[str, Any]], outpath: Path) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception as exc:
        print(f"skip exp5 Pillow figure: {exc}")
        return

    rows = rows[:10]
    labels = [f"{r['clip']}:{r['start_frame']}" for r in rows]
    l1_scores = [float(r["score_rmse"]) for r in rows]
    g_scores = [float(r["gaussian_rmse"]) for r in rows]
    l1_q95 = float(get_path(payload, "l1_tpca.test_rmse_q95"))
    g_q95 = float(get_path(payload, "gaussian_tpca.test_rmse_q95"))

    width, height = 1600, 900
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
        small = ImageFont.truetype("arial.ttf", 18)
        title_font = ImageFont.truetype("arial.ttf", 34)
    except OSError:
        font = small = title_font = ImageFont.load_default()

    draw.text((60, 28), "Experiment 5: Top L1-TPCA Video Anomaly Windows", fill="#222222", font=title_font)

    def panel(y0: int, y1: int, vals: list[float], q95: float, ylabel: str, color: str) -> None:
        left, right = 120, width - 60
        top, bottom = y0 + 35, y1 - 85
        vmax = max(vals + [q95]) * 1.08
        draw.line((left, bottom, right, bottom), fill="#333333", width=2)
        draw.line((left, top, left, bottom), fill="#333333", width=2)
        draw.text((left, y0), ylabel, fill="#222222", font=font)
        draw.text((left + 520, y0 + 4), "dashed line = 95% test RMSE", fill="#555555", font=small)

        qy = bottom - (q95 / vmax) * (bottom - top)
        draw.line((left, qy, right, qy), fill="#d95f02", width=3)
        for dash in range(left, right, 28):
            draw.line((dash, qy, min(dash + 14, right), qy), fill="#d95f02", width=5)

        gap = (right - left) / len(vals)
        bar_w = gap * 0.58
        for i, val in enumerate(vals):
            cx = left + gap * i + gap / 2
            x0 = cx - bar_w / 2
            x1 = cx + bar_w / 2
            y = bottom - (val / vmax) * (bottom - top)
            draw.rectangle((x0, y, x1, bottom), fill=color)
            draw.text((x0, y - 24), f"{val:.3g}", fill="#222222", font=small)
        draw.text((left - 86, top - 6), f"{vmax:.3g}", fill="#555555", font=small)
        draw.text((left - 72, bottom - 12), "0", fill="#555555", font=small)

    panel(95, 430, l1_scores, l1_q95, "L1-TPCA window RMSE", "#2c7fb8")
    panel(470, 810, g_scores, g_q95, "Gaussian TPCA window RMSE", "#7f3b08")

    left, right = 120, width - 60
    gap = (right - left) / len(labels)
    for i, lab in enumerate(labels):
        x = left + gap * i + 6
        draw.text((x, 815), lab, fill="#333333", font=small)

    img.save(outpath)


def require_matplotlib():
    if CSV_ONLY:
        raise RuntimeError("csv_only")
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Matplotlib is required for figure generation. The script still writes CSV summaries; "
            "install matplotlib or run this in an environment that has it."
        ) from exc
    return plt


def save_line_plot(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_keys: list[str],
    labels: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    panel_key: str | None = None,
) -> None:
    try:
        plt = require_matplotlib()
    except RuntimeError as exc:
        print(f"skip figure {path.name}: {exc}")
        return
    panels = sorted({r[panel_key] for r in rows}) if panel_key else [None]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 3.6), squeeze=False)
    for ax, panel in zip(axes.ravel(), panels):
        sub = [r for r in rows if panel is None or r[panel_key] == panel]
        xs = sorted({r[x_key] for r in sub})
        for y_key, label in zip(y_keys, labels):
            means = []
            ses = []
            for x in xs:
                vals = np.asarray([float(r[y_key]) for r in sub if r[x_key] == x], dtype=float)
                vals = vals[np.isfinite(vals)]
                means.append(float(vals.mean()) if vals.size else np.nan)
                ses.append(float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0)
            ax.errorbar(xs, means, yerr=ses, marker="o", linewidth=1.8, capsize=3, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}" if panel is None else f"{title}: {panel_key}={panel}")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_boxplot(
    data: list[np.ndarray],
    labels: list[str],
    *,
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    try:
        plt = require_matplotlib()
    except RuntimeError as exc:
        print(f"skip figure {path.name}: {exc}")
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_exp1(path: Path, outdir: Path) -> None:
    payload = load_json(path)
    rows = payload["rows"]
    flat = []
    for r in rows:
        flat.append(
            {
                "N": r["N"],
                "rho_true": r["rho_true"],
                "l1_err": get_path(r, "l1_tpca.subspace_error.Err"),
                "gaussian_err": get_path(r, "gaussian_tpca.subspace_error.Err"),
                "l1_rmse": get_path(r, "l1_tpca.rmse"),
                "gaussian_rmse": get_path(r, "gaussian_tpca.rmse"),
            }
        )
    write_rows_csv(outdir / "exp1_flat.csv", flat)
    write_rows_csv(
        outdir / "exp1_summary.csv",
        grouped_summary(
            flat,
            ["N", "rho_true"],
            {
                "l1_err": lambda r: r["l1_err"],
                "gaussian_err": lambda r: r["gaussian_err"],
                "l1_rmse": lambda r: r["l1_rmse"],
                "gaussian_rmse": lambda r: r["gaussian_rmse"],
            },
        ),
    )
    save_line_plot(
        flat,
        x_key="N",
        y_keys=["l1_err", "gaussian_err"],
        labels=["L1-TPCA", "Gaussian TPCA"],
        title="Subspace Recovery",
        xlabel="Sample size N",
        ylabel="Mean subspace error",
        panel_key="rho_true",
        path=outdir / "exp1_subspace_error_vs_N.png",
    )
    save_line_plot(
        flat,
        x_key="N",
        y_keys=["l1_rmse", "gaussian_rmse"],
        labels=["L1-TPCA", "Gaussian TPCA"],
        title="Reconstruction",
        xlabel="Sample size N",
        ylabel="RMSE",
        panel_key="rho_true",
        path=outdir / "exp1_rmse_vs_N.png",
    )


def plot_exp2(path: Path, outdir: Path) -> None:
    payload = load_json(path)
    rows = payload["rows"]
    flat = []
    clean_weights = []
    outlier_weights = []
    clean_delta = []
    outlier_delta = []
    for r in rows:
        weights = np.asarray(get_path(r, "l1_tpca.weights", []), dtype=float)
        deltas = np.asarray(get_path(r, "l1_tpca.delta", []), dtype=float)
        flat.append(
            {
                "contamination_type": r["contamination_type"],
                "rate": r["rate"],
                "outlier_scale": r["outlier_scale"],
                "l1_err": get_path(r, "l1_tpca.subspace_error.Err"),
                "gaussian_err": get_path(r, "gaussian_tpca.subspace_error.Err"),
                "l1_rmse": get_path(r, "l1_tpca.rmse"),
                "gaussian_rmse": get_path(r, "gaussian_tpca.rmse"),
                "weight_clean_mean": get_path(r, "l1_tpca.weight_clean_mean"),
                "weight_outlier_mean": get_path(r, "l1_tpca.weight_outlier_mean"),
                "delta_clean_mean": get_path(r, "l1_tpca.delta_clean_mean"),
                "delta_outlier_mean": get_path(r, "l1_tpca.delta_outlier_mean"),
            }
        )
        wc = as_float(get_path(r, "l1_tpca.weight_clean_mean"))
        wo = as_float(get_path(r, "l1_tpca.weight_outlier_mean"))
        dc = as_float(get_path(r, "l1_tpca.delta_clean_mean"))
        do = as_float(get_path(r, "l1_tpca.delta_outlier_mean"))
        if np.isfinite(wc):
            clean_weights.append(wc)
        if np.isfinite(wo):
            outlier_weights.append(wo)
        if np.isfinite(dc):
            clean_delta.append(dc)
        if np.isfinite(do):
            outlier_delta.append(do)
    write_rows_csv(outdir / "exp2_flat.csv", flat)
    write_rows_csv(
        outdir / "exp2_summary.csv",
        grouped_summary(
            flat,
            ["contamination_type", "rate", "outlier_scale"],
            {
                "l1_err": lambda r: r["l1_err"],
                "gaussian_err": lambda r: r["gaussian_err"],
                "weight_clean_mean": lambda r: r["weight_clean_mean"],
                "weight_outlier_mean": lambda r: r["weight_outlier_mean"],
            },
        ),
    )
    save_line_plot(
        flat,
        x_key="rate",
        y_keys=["l1_err", "gaussian_err"],
        labels=["L1-TPCA", "Gaussian TPCA"],
        title="Contamination Robustness",
        xlabel="Contamination rate",
        ylabel="Mean subspace error",
        panel_key="contamination_type",
        path=outdir / "exp2_error_vs_contamination_rate.png",
    )
    if clean_weights and outlier_weights:
        save_boxplot(
            [np.asarray(clean_weights), np.asarray(outlier_weights)],
            ["clean", "outlier"],
            title="Posterior Weight Diagnostic",
            ylabel="Mean posterior weight",
            path=outdir / "exp2_weight_boxplot.png",
        )
    if clean_delta and outlier_delta:
        save_boxplot(
            [np.asarray(clean_delta), np.asarray(outlier_delta)],
            ["clean", "outlier"],
            title="Residual Proxy Diagnostic",
            ylabel="Mean delta",
            path=outdir / "exp2_delta_boxplot.png",
        )


def plot_exp3(path: Path, outdir: Path) -> None:
    payload = load_json(path)
    rows = payload["rows"]
    flat = []
    for r in rows:
        flat.append(
            {
                "method": r["method"],
                "N": r["N"],
                "dimension": int(np.prod(r["shape"])),
                "seconds": r["seconds"],
                "n_iter": r["n_iter"],
            }
        )
    write_rows_csv(outdir / "exp3_flat.csv", flat)
    write_rows_csv(
        outdir / "exp3_summary.csv",
        grouped_summary(flat, ["method", "N", "dimension"], {"seconds": lambda r: r["seconds"], "n_iter": lambda r: r["n_iter"]}),
    )
    methods = sorted({r["method"] for r in flat})
    for x_key, x_label, fname in [
        ("N", "Sample size N", "exp3_runtime_vs_N.png"),
        ("dimension", "Tensor dimension", "exp3_runtime_vs_dimension.png"),
    ]:
        rows2 = []
        for r in flat:
            rec = {x_key: r[x_key]}
            for m in methods:
                rec[m] = r["seconds"] if r["method"] == m else np.nan
            rows2.append(rec)
        save_line_plot(
            rows2,
            x_key=x_key,
            y_keys=methods,
            labels=methods,
            title="Runtime Scaling",
            xlabel=x_label,
            ylabel="Seconds",
            path=outdir / fname,
        )


def plot_exp4(path: Path, outdir: Path) -> None:
    payload = load_json(path)
    rows = [
        {
            "method": "L1-TPCA",
            "train_rmse": get_path(payload, "l1_tpca.train_rmse"),
            "test_rmse": get_path(payload, "l1_tpca.test_rmse"),
            "accuracy": get_path(payload, "l1_tpca.nearest_centroid_accuracy"),
        },
        {
            "method": "Gaussian TPCA",
            "train_rmse": get_path(payload, "gaussian_tpca.train_rmse"),
            "test_rmse": get_path(payload, "gaussian_tpca.test_rmse"),
            "accuracy": get_path(payload, "gaussian_tpca.nearest_centroid_accuracy"),
        },
    ]
    write_rows_csv(outdir / f"exp4_{payload['config']['dataset']}_summary.csv", rows)
    try:
        plt = require_matplotlib()
    except RuntimeError as exc:
        print(f"skip exp4 figure: {exc}")
        return
    methods = [r["method"] for r in rows]
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
    axes[0].bar(x - 0.18, [r["train_rmse"] for r in rows], width=0.36, label="train")
    axes[0].bar(x + 0.18, [r["test_rmse"] for r in rows], width=0.36, label="test")
    axes[0].set_xticks(x, methods, rotation=15)
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Hyperspectral Reconstruction")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, [r["accuracy"] for r in rows], width=0.5)
    axes[1].set_xticks(x, methods, rotation=15)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Latent Feature Classification")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / f"exp4_{payload['config']['dataset']}_bars.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_exp5(path: Path, outdir: Path) -> None:
    payload = load_json(path)
    summary_rows = [
        {
            "method": "L1-TPCA",
            "train_rmse": get_path(payload, "l1_tpca.train_rmse"),
            "test_rmse_mean": get_path(payload, "l1_tpca.test_rmse_mean"),
            "test_rmse_q95": get_path(payload, "l1_tpca.test_rmse_q95"),
        },
        {
            "method": "Gaussian TPCA",
            "train_rmse": get_path(payload, "gaussian_tpca.train_rmse"),
            "test_rmse_mean": get_path(payload, "gaussian_tpca.test_rmse_mean"),
            "test_rmse_q95": get_path(payload, "gaussian_tpca.test_rmse_q95"),
        },
    ]
    write_rows_csv(outdir / "exp5_summary.csv", summary_rows)
    rows = payload.get("top_l1_test_windows", [])
    write_rows_csv(outdir / "exp5_top_windows.csv", rows)
    if not rows:
        return
    try:
        plt = require_matplotlib()
    except RuntimeError as exc:
        print(f"skip exp5 figure: {exc}")
        if not CSV_ONLY:
            plot_exp5_pillow(payload, rows, outdir / "exp5_top_video_windows.png")
        return
    labels = [f"{r['clip']}:{r['start_frame']}" for r in rows[:10]]
    l1_scores = [r["score_rmse"] for r in rows[:10]]
    g_scores = [r["gaussian_rmse"] for r in rows[:10]]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.8), sharex=True)
    axes[0].bar(x, l1_scores, width=0.58, color="#2c7fb8")
    axes[0].axhline(get_path(payload, "l1_tpca.test_rmse_q95"), color="#d95f02", linewidth=1.2, linestyle="--", label="95% test RMSE")
    axes[0].set_ylabel("L1-TPCA RMSE")
    axes[0].set_title("Top L1-TPCA Video Anomaly Windows")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, g_scores, width=0.58, color="#7f3b08")
    axes[1].axhline(get_path(payload, "gaussian_tpca.test_rmse_q95"), color="#d95f02", linewidth=1.2, linestyle="--", label="95% test RMSE")
    axes[1].set_xticks(x, labels, rotation=40, ha="right")
    axes[1].set_ylabel("Gaussian RMSE")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "exp5_top_video_windows.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert L1-TPCA JSON experiment outputs into paper figures and CSV summaries.")
    ap.add_argument("--exp1", type=str, default="results/exp1_model_specification.json")
    ap.add_argument("--exp2", type=str, default="results/exp2_contamination_robustness.json")
    ap.add_argument("--exp3", type=str, default="results/exp3_computation_scalability.json")
    ap.add_argument("--exp4", type=str, nargs="*", default=["results/exp4_hyperspectral_realdata.json"])
    ap.add_argument("--exp5", type=str, default="results/exp5_video_anomaly_realdata.json")
    ap.add_argument("--outdir", type=str, default="figures")
    ap.add_argument("--skip_missing", action="store_true")
    ap.add_argument("--csv_only", action="store_true", help="Write CSV summaries without creating PNG figures.")
    args = ap.parse_args()

    global CSV_ONLY
    CSV_ONLY = bool(args.csv_only)

    outdir = ensure_dir(args.outdir)
    jobs = [
        ("exp1", args.exp1, plot_exp1),
        ("exp2", args.exp2, plot_exp2),
        ("exp3", args.exp3, plot_exp3),
        ("exp5", args.exp5, plot_exp5),
    ]
    for name, path, func in jobs:
        p = Path(path)
        if not p.exists():
            if args.skip_missing:
                print(f"skip missing {name}: {p}")
                continue
            raise FileNotFoundError(p)
        print(f"plot {name}: {p}")
        func(p, outdir)

    for ptext in args.exp4:
        p = Path(ptext)
        if not p.exists():
            if args.skip_missing:
                print(f"skip missing exp4: {p}")
                continue
            raise FileNotFoundError(p)
        print(f"plot exp4: {p}")
        plot_exp4(p, outdir)


if __name__ == "__main__":
    main()
