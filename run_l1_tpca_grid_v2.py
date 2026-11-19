from __future__ import annotations

import argparse
import csv
import datetime as dt
import itertools
import json
import math
import os
from collections import defaultdict
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from l1_tpca_contamination_v2 import run_one_experiment


SCENARIOS = ["gaussian_clean", "laplace_clean", "tensor_outliers", "slice_outliers", "hybrid"]


def parse_tuple_int(s: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def parse_list_str(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_list_float(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_list_int(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def resolve_lam_init(sigma2: float, lam: float | None) -> float:
    return float(2.0 / max(sigma2, 1e-12) if lam is None else lam)


def prior_scheme_values(scheme: str, lam_init: float) -> Tuple[float, float]:
    scheme = scheme.lower()
    if scheme == "fixed":
        return 0.0, 0.0
    if scheme == "none":
        return 0.0, 0.0
    if scheme == "mild":
        return 5.0, 5.0 / max(lam_init, 1e-12)
    if scheme == "strong":
        return 20.0, 20.0 / max(lam_init, 1e-12)
    if scheme == "very_strong":
        return 50.0, 50.0 / max(lam_init, 1e-12)
    raise ValueError(f"Unknown prior scheme: {scheme}")


def flatten(obj: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten(v, new_prefix))
    elif isinstance(obj, (list, tuple)):
        if all(not isinstance(v, (dict, list, tuple)) for v in obj):
            for i, v in enumerate(obj):
                out[f"{prefix}.{i}" if prefix else str(i)] = v
            out[prefix] = json.dumps(list(obj)) if prefix else json.dumps(list(obj))
        else:
            out[prefix] = json.dumps(obj)
    else:
        out[prefix] = obj
    return out


RAW_SKIP_FIELDS = {
    "shape",
    "ranks",
}


CONFIG_KEYS = [
    "shape",
    "ranks",
    "scenario",
    "N",
    "sigma2",
    "contamination_rate",
    "outlier_scale",
    "lam",
    "lam_init",
    "slice_mode",
    "vbem_iter",
    "power_iter",
    "gaussian_iter",
    "device",
    "update_lam",
    "z_prior_precision",
    "prior_scheme",
    "lam_prior_a",
    "lam_prior_b",
    "mstep_inner",
    "weight_damping",
    "run_two_stage",
    "two_stage_blend",
]


SUMMARY_METRICS = [
    "robust_vbem.Err",
    "robust_power.Err",
    "gaussian_em.Err",
    "two_stage_vbem.Err",
    "robust_vbem.weight_summary.topk_low_score_precision",
    "robust_vbem.weight_summary.topk_low_score_recall",
    "robust_vbem.delta_summary.topk_high_score_precision",
    "robust_vbem.delta_summary.topk_high_score_recall",
    "two_stage_vbem.weight_summary.topk_low_score_precision",
    "two_stage_vbem.weight_summary.topk_low_score_recall",
    "two_stage_vbem.delta_summary.topk_high_score_precision",
    "two_stage_vbem.delta_summary.topk_high_score_recall",
    "robust_power.weight_summary.topk_low_score_precision",
    "robust_power.weight_summary.topk_low_score_recall",
    "robust_vbem.weight_summary.outlier_to_clean_mean_ratio",
    "robust_vbem.delta_summary.outlier_to_clean_mean_ratio",
    "two_stage_vbem.weight_summary.outlier_to_clean_mean_ratio",
    "two_stage_vbem.delta_summary.outlier_to_clean_mean_ratio",
    "gaussian_em.sigma2_hat",
    "robust_vbem.n_iter",
    "two_stage_vbem.n_iter",
    "robust_vbem.converged",
    "two_stage_vbem.converged",
    "gaussian_em.converged",
]


def make_row(config: Dict[str, Any], result: Dict[str, Any], seed: int) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "seed": seed,
        "status": "ok",
    }
    for k, v in config.items():
        if isinstance(v, (tuple, list)):
            row[f"config.{k}"] = ",".join(str(x) for x in v)
        else:
            row[f"config.{k}"] = v
    flat = flatten(result)
    for k, v in flat.items():
        if k in RAW_SKIP_FIELDS:
            continue
        row[k] = v
    return row


def make_error_row(config: Dict[str, Any], seed: int, exc: Exception) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "seed": seed,
        "status": "error",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }
    for k, v in config.items():
        if isinstance(v, (tuple, list)):
            row[f"config.{k}"] = ",".join(str(x) for x in v)
        else:
            row[f"config.{k}"] = v
    return row


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def try_float(v: Any) -> float | None:
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except Exception:
        return None



def build_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(f"config.{k}") for k in CONFIG_KEYS)
        groups[key].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for key, group_rows in groups.items():
        summary: Dict[str, Any] = {f"config.{k}": val for k, val in zip(CONFIG_KEYS, key)}
        summary["n_runs"] = len(group_rows)
        summary["n_success"] = sum(1 for r in group_rows if r.get("status") == "ok")
        summary["n_error"] = sum(1 for r in group_rows if r.get("status") == "error")
        ok_rows = [r for r in group_rows if r.get("status") == "ok"]
        if not ok_rows:
            summary_rows.append(summary)
            continue
        for metric in SUMMARY_METRICS:
            vals = [try_float(r.get(metric)) for r in ok_rows]
            vals = [v for v in vals if v is not None and math.isfinite(v)]
            if vals:
                summary[f"{metric}.mean"] = mean(vals)
                summary[f"{metric}.median"] = median(vals)
                summary[f"{metric}.min"] = min(vals)
                summary[f"{metric}.max"] = max(vals)
        summary_rows.append(summary)
    return summary_rows



def generate_configs(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    scenarios = parse_list_str(args.scenarios)
    update_lam_values = [bool(x) for x in parse_list_int(args.update_lam_values)]
    z_priors = parse_list_float(args.z_prior_precisions)
    mstep_inners = parse_list_int(args.mstep_inner_values)
    dampings = parse_list_float(args.weight_damping_values)
    run_two_stage_values = [bool(x) for x in parse_list_int(args.run_two_stage_values)]
    blend_values = parse_list_float(args.two_stage_blend_values)
    prior_schemes = parse_list_str(args.prior_schemes)

    lam_init = resolve_lam_init(args.sigma2, args.lam)

    for scenario, update_lam, z_prec, mstep_inner, damping, run_two_stage, blend in itertools.product(
        scenarios,
        update_lam_values,
        z_priors,
        mstep_inners,
        dampings,
        run_two_stage_values,
        blend_values,
    ):
        schemes = prior_schemes if update_lam else ["fixed"]
        for scheme in schemes:
            a, b = prior_scheme_values(scheme, lam_init)
            yield {
                "shape": args.shape,
                "ranks": args.ranks,
                "scenario": scenario,
                "N": args.N,
                "sigma2": args.sigma2,
                "contamination_rate": args.contamination_rate,
                "outlier_scale": args.outlier_scale,
                "lam": args.lam,
                "lam_init": lam_init,
                "slice_mode": args.slice_mode,
                "vbem_iter": args.vbem_iter,
                "power_iter": args.power_iter,
                "gaussian_iter": args.gaussian_iter,
                "device": args.device,
                "update_lam": update_lam,
                "z_prior_precision": z_prec,
                "prior_scheme": scheme,
                "lam_prior_a": a,
                "lam_prior_b": b,
                "mstep_inner": mstep_inner,
                "weight_damping": damping,
                "run_two_stage": run_two_stage,
                "two_stage_blend": blend,
            }



def run_grid(args: argparse.Namespace) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    configs = list(generate_configs(args))
    total = len(configs) * args.replications
    counter = 0

    for cfg in configs:
        for rep in range(args.replications):
            seed = args.seed + rep
            counter += 1
            if args.verbose:
                print(f"[{counter}/{total}] scenario={cfg['scenario']} update_lam={cfg['update_lam']} z_prior={cfg['z_prior_precision']} prior={cfg['prior_scheme']} mstep_inner={cfg['mstep_inner']} damping={cfg['weight_damping']} two_stage={cfg['run_two_stage']} blend={cfg['two_stage_blend']} seed={seed}")
            try:
                result = run_one_experiment(
                    shape=cfg["shape"],
                    ranks=cfg["ranks"],
                    scenario=cfg["scenario"],
                    N=cfg["N"],
                    sigma2=cfg["sigma2"],
                    seed=seed,
                    contamination_rate=cfg["contamination_rate"],
                    outlier_scale=cfg["outlier_scale"],
                    lam=cfg["lam"],
                    slice_mode=cfg["slice_mode"],
                    vbem_iter=cfg["vbem_iter"],
                    power_iter=cfg["power_iter"],
                    gaussian_iter=cfg["gaussian_iter"],
                    device=cfg["device"],
                    update_lam=cfg["update_lam"],
                    mstep_inner=cfg["mstep_inner"],
                    z_prior_precision=cfg["z_prior_precision"],
                    lam_prior_a=cfg["lam_prior_a"],
                    lam_prior_b=cfg["lam_prior_b"],
                    weight_damping=cfg["weight_damping"],
                    run_two_stage=cfg["run_two_stage"],
                    two_stage_blend=cfg["two_stage_blend"],
                )
                rows.append(make_row(cfg, result, seed))
            except Exception as exc:
                rows.append(make_error_row(cfg, seed, exc))
                if args.fail_fast:
                    raise

    summary_rows = build_summary(rows)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.output_prefix or f"l1_tpca_grid_v2_{ts}"
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    raw_csv = os.path.join(out_dir, f"{prefix}_raw.csv")
    summary_csv = os.path.join(out_dir, f"{prefix}_summary.csv")
    manifest_json = os.path.join(out_dir, f"{prefix}_manifest.json")

    write_csv(raw_csv, rows)
    write_csv(summary_csv, summary_rows)

    manifest = {
        "created_at": ts,
        "raw_csv": raw_csv,
        "summary_csv": summary_csv,
        "n_configurations": len(configs),
        "replications": args.replications,
        "n_rows": len(rows),
        "n_summary_rows": len(summary_rows),
        "grid": {
            "scenarios": parse_list_str(args.scenarios),
            "update_lam_values": [bool(x) for x in parse_list_int(args.update_lam_values)],
            "z_prior_precisions": parse_list_float(args.z_prior_precisions),
            "prior_schemes": parse_list_str(args.prior_schemes),
            "mstep_inner_values": parse_list_int(args.mstep_inner_values),
            "weight_damping_values": parse_list_float(args.weight_damping_values),
            "run_two_stage_values": [bool(x) for x in parse_list_int(args.run_two_stage_values)],
            "two_stage_blend_values": parse_list_float(args.two_stage_blend_values),
        },
        "fixed": {
            "shape": args.shape,
            "ranks": args.ranks,
            "N": args.N,
            "sigma2": args.sigma2,
            "contamination_rate": args.contamination_rate,
            "outlier_scale": args.outlier_scale,
            "lam": args.lam,
            "slice_mode": args.slice_mode,
            "vbem_iter": args.vbem_iter,
            "power_iter": args.power_iter,
            "gaussian_iter": args.gaussian_iter,
            "device": args.device,
            "seed_base": args.seed,
        },
    }
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run a version-2 L1-TPCA experiment grid and save raw/summary CSV files.")
    ap.add_argument("--shape", type=parse_tuple_int, default=(6, 6, 6), help="Tensor shape, e.g. 6,6,6")
    ap.add_argument("--ranks", type=parse_tuple_int, default=(2, 2, 2), help="Tucker ranks, e.g. 2,2,2")
    ap.add_argument("--scenarios", type=str, default=",".join(SCENARIOS), help="Comma-separated scenarios.")
    ap.add_argument("--N", type=int, default=60)
    ap.add_argument("--sigma2", type=float, default=1.0)
    ap.add_argument("--contamination_rate", type=float, default=0.1)
    ap.add_argument("--outlier_scale", type=float, default=8.0)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--slice_mode", type=int, default=0)
    ap.add_argument("--replications", type=int, default=3, help="Replications per hyperparameter setting.")
    ap.add_argument("--seed", type=int, default=0, help="Base seed; each replication uses seed+rep.")
    ap.add_argument("--vbem_iter", type=int, default=200)
    ap.add_argument("--power_iter", type=int, default=50)
    ap.add_argument("--gaussian_iter", type=int, default=40)
    ap.add_argument("--update_lam_values", type=str, default="0,1", help="Comma-separated 0/1 values.")
    ap.add_argument("--z_prior_precisions", type=str, default="1.0,5.0,10.0,50.0")
    ap.add_argument("--prior_schemes", type=str, default="none,mild,strong", help="Used only when update_lam=1. fixed is used automatically when update_lam=0.")
    ap.add_argument("--mstep_inner_values", type=str, default="5", help="Comma-separated integers.")
    ap.add_argument("--weight_damping_values", type=str, default="0.0", help="Comma-separated floats in [0,1].")
    ap.add_argument("--run_two_stage_values", type=str, default="1", help="Comma-separated 0/1 values.")
    ap.add_argument("--two_stage_blend_values", type=str, default="1.0", help="Comma-separated floats.")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--output_dir", type=str, default="/mnt/data")
    ap.add_argument("--output_prefix", type=str, default="")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--fail_fast", action="store_true")
    args = ap.parse_args()

    manifest = run_grid(args)
    print(json.dumps(manifest, indent=2))
