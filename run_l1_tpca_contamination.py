from __future__ import annotations

import argparse
import json

from l1_tpca_contamination import run_one_experiment, summarize_results


def parse_shape(s: str):
    return tuple(int(x) for x in s.split(","))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Contamination experiments for L1-TPCA vs Gaussian TPCA.")
    ap.add_argument("--shape", type=parse_shape, default="6,6,6", help="Tensor shape, e.g. 6,6,6")
    ap.add_argument("--ranks", type=parse_shape, default="2,2,2", help="Tucker ranks, e.g. 2,2,2")
    ap.add_argument("--scenario", type=str, default="tensor_outliers",
                    choices=["gaussian_clean", "laplace_clean", "tensor_outliers", "slice_outliers", "hybrid"],
                    help="Data scenario to test.")
    ap.add_argument("--N", type=int, default=60, help="Number of samples.")
    ap.add_argument("--sigma2", type=float, default=1.0, help="Baseline Gaussian noise variance.")
    ap.add_argument("--contamination_rate", type=float, default=0.1, help="Fraction of contaminated samples.")
    ap.add_argument("--outlier_scale", type=float, default=8.0, help="Outlier-noise standard deviation multiplier over sqrt(sigma2).")
    ap.add_argument("--lam", type=float, default=None, help="Laplace scale-mixture rate. Default matches E[tau]=sigma2.")
    ap.add_argument("--slice_mode", type=int, default=0, help="Mode along which a slice is corrupted for slice_outliers/hybrid.")
    ap.add_argument("--replications", type=int, default=10, help="Number of random replications.")
    ap.add_argument("--seed", type=int, default=0, help="Base seed.")
    ap.add_argument("--vbem_iter", type=int, default=200)
    ap.add_argument("--power_iter", type=int, default=50)
    ap.add_argument("--gaussian_iter", type=int, default=100)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    results = []
    for rep in range(args.replications):
        results.append(
            run_one_experiment(
                shape=args.shape,
                ranks=args.ranks,
                scenario=args.scenario,
                N=args.N,
                sigma2=args.sigma2,
                seed=args.seed + rep,
                contamination_rate=args.contamination_rate,
                outlier_scale=args.outlier_scale,
                lam=args.lam,
                slice_mode=args.slice_mode,
                vbem_iter=args.vbem_iter,
                power_iter=args.power_iter,
                gaussian_iter=args.gaussian_iter,
                device=args.device,
            )
        )

    payload = {
        "config": vars(args),
        "summary": summarize_results(results),
        "results": results,
    }
    print(json.dumps(payload, indent=2))
