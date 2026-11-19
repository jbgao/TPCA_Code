from __future__ import annotations

import argparse
import time

import numpy as np

from exp_l1_tpca_common import fit_l1_and_gaussian, parse_shape, write_json
from l1_tpca_numpy import random_factors, sample_l1_tpca


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Experiment 1: correctly specified Laplace scale-mixture TPCA.")
    ap.add_argument("--shape", type=parse_shape, default="5,5,4")
    ap.add_argument("--ranks", type=parse_shape, default="2,2,2")
    ap.add_argument("--sample_sizes", type=parse_shape, default="20,40")
    ap.add_argument("--rhos", type=str, default="0.75,2.0,6.0")
    ap.add_argument("--replications", type=int, default=5)
    ap.add_argument("--max_iter", type=int, default=150)
    ap.add_argument("--mstep_inner", type=int, default=5)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--output", type=str, default="results/exp1_model_specification.json")
    args = ap.parse_args()

    rows = []
    for N in args.sample_sizes:
        for rho in [float(x) for x in args.rhos.split(",")]:
            for rep in range(args.replications):
                seed = args.seed + 1000 * rep + 10 * N + int(100 * rho)
                rng = np.random.default_rng(seed)
                factors = random_factors(args.shape, args.ranks, rng)
                samples, _, _ = sample_l1_tpca(factors, rho, N, rng)
                start = time.perf_counter()
                metrics = fit_l1_and_gaussian(
                    samples,
                    factors,
                    args.ranks,
                    max_iter=args.max_iter,
                    mstep_inner=args.mstep_inner,
                    seed=seed,
                    rho_init=max(rho, 1e-3),
                )
                rows.append(
                    {
                        "N": int(N),
                        "rho_true": float(rho),
                        "rep": int(rep),
                        "seconds": float(time.perf_counter() - start),
                        **metrics,
                    }
                )

    write_json({"config": vars(args), "rows": rows}, args.output)
