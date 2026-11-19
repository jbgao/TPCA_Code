from __future__ import annotations

import argparse
import time

import numpy as np

from exp_l1_tpca_common import fit_l1_and_gaussian, parse_shape, write_json
from l1_tpca_numpy import contaminate_tensor_samples, random_factors, sample_l1_tpca


def contaminate_slices(samples, rng, *, contamination_rate: float, outlier_scale: float, slice_mode: int = 0):
    out = [np.asarray(X, dtype=float).copy() for X in samples]
    n_out = max(1, int(round(contamination_rate * len(out)))) if contamination_rate > 0 else 0
    mask = np.zeros(len(out), dtype=bool)
    if n_out == 0:
        return out, mask
    idx = rng.choice(len(out), size=n_out, replace=False)
    for i in idx:
        mask[i] = True
        sl = [slice(None)] * out[i].ndim
        sl[slice_mode] = int(rng.integers(0, out[i].shape[slice_mode]))
        out[i][tuple(sl)] += rng.normal(scale=outlier_scale, size=out[i][tuple(sl)].shape)
    return out, mask


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Experiment 2: robustness under gross tensor and slice contamination.")
    ap.add_argument("--shape", type=parse_shape, default="5,5,4")
    ap.add_argument("--ranks", type=parse_shape, default="2,2,2")
    ap.add_argument("--N", type=int, default=50)
    ap.add_argument("--rho", type=float, default=2.0)
    ap.add_argument("--rates", type=str, default="0,0.05,0.1,0.2")
    ap.add_argument("--outlier_scales", type=str, default="4,8,12")
    ap.add_argument("--types", type=str, default="tensor,slice")
    ap.add_argument("--replications", type=int, default=5)
    ap.add_argument("--max_iter", type=int, default=40)
    ap.add_argument("--mstep_inner", type=int, default=2)
    ap.add_argument("--seed", type=int, default=200)
    ap.add_argument("--output", type=str, default="results/exp2_contamination_robustness.json")
    args = ap.parse_args()

    rows = []
    for ctype in args.types.split(","):
        for rate in [float(x) for x in args.rates.split(",")]:
            for scale in [float(x) for x in args.outlier_scales.split(",")]:
                for rep in range(args.replications):
                    seed = args.seed + rep * 10000 + int(1000 * rate) + int(10 * scale)
                    rng = np.random.default_rng(seed)
                    factors = random_factors(args.shape, args.ranks, rng)
                    clean_samples, _, _ = sample_l1_tpca(factors, args.rho, args.N, rng)
                    if ctype == "tensor":
                        samples, mask = contaminate_tensor_samples(clean_samples, rng, contamination_rate=rate, outlier_scale=scale)
                    elif ctype == "slice":
                        samples, mask = contaminate_slices(clean_samples, rng, contamination_rate=rate, outlier_scale=scale)
                    else:
                        raise ValueError(f"unknown contamination type {ctype!r}")
                    start = time.perf_counter()
                    metrics = fit_l1_and_gaussian(
                        samples,
                        factors,
                        args.ranks,
                        max_iter=args.max_iter,
                        mstep_inner=args.mstep_inner,
                        seed=seed,
                        rho_init=args.rho,
                    )
                    weights = np.asarray(metrics["l1_tpca"].get("weights", []), dtype=float)
                    deltas = np.asarray(metrics["l1_tpca"].get("delta", []), dtype=float)
                    if weights.size:
                        metrics["l1_tpca"]["weight_clean_mean"] = float(weights[~mask].mean()) if (~mask).sum() else None
                        metrics["l1_tpca"]["weight_outlier_mean"] = float(weights[mask].mean()) if mask.sum() else None
                    if deltas.size:
                        metrics["l1_tpca"]["delta_clean_mean"] = float(deltas[~mask].mean()) if (~mask).sum() else None
                        metrics["l1_tpca"]["delta_outlier_mean"] = float(deltas[mask].mean()) if mask.sum() else None
                    rows.append(
                        {
                            "contamination_type": ctype,
                            "rate": float(rate),
                            "outlier_scale": float(scale),
                            "rep": int(rep),
                            "outlier_count": int(mask.sum()),
                            "seconds": float(time.perf_counter() - start),
                            **metrics,
                        }
                    )

    write_json({"config": vars(args), "rows": rows}, args.output)
