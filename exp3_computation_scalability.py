from __future__ import annotations

import argparse
import time

import numpy as np

from exp_l1_tpca_common import parse_shape, write_json
from l1_tpca_numpy import GaussianTPCAALS, L1TPCAVBEM, random_factors, sample_l1_tpca


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Experiment 3: runtime and convergence scaling.")
    ap.add_argument("--shapes", type=str, default="4,4,4;6,6,4;8,8,4")
    ap.add_argument("--ranks", type=parse_shape, default="2,2,2")
    ap.add_argument("--N_values", type=parse_shape, default="20,40")
    ap.add_argument("--rho", type=float, default=2.0)
    ap.add_argument("--replications", type=int, default=3)
    ap.add_argument("--max_iter", type=int, default=20)
    ap.add_argument("--mstep_inner", type=int, default=1)
    ap.add_argument("--seed", type=int, default=300)
    ap.add_argument("--output", type=str, default="results/exp3_computation_scalability.json")
    args = ap.parse_args()

    rows = []
    for shape in [parse_shape(s) for s in args.shapes.split(";")]:
        for N in args.N_values:
            for rep in range(args.replications):
                seed = args.seed + 1000 * rep + int(np.prod(shape)) + N
                rng = np.random.default_rng(seed)
                factors = random_factors(shape, args.ranks, rng)
                samples, _, _ = sample_l1_tpca(factors, args.rho, N, rng)

                for exact in [False, True]:
                    start = time.perf_counter()
                    fit = L1TPCAVBEM(
                        args.ranks,
                        rho_init=args.rho,
                        max_iter=args.max_iter,
                        mstep_inner=args.mstep_inner,
                        exact_second_moment=exact,
                        seed=seed,
                    ).fit(samples)
                    rows.append(
                        {
                            "method": "l1_tpca_exact_mstep" if exact else "l1_tpca_ecm_mstep",
                            "shape": shape,
                            "N": int(N),
                            "rep": int(rep),
                            "seconds": float(time.perf_counter() - start),
                            "n_iter": int(fit.n_iter),
                            "converged": bool(fit.converged),
                            "rho_hat": float(fit.rho),
                        }
                    )

                start = time.perf_counter()
                gfit = GaussianTPCAALS(
                    args.ranks,
                    max_iter=args.max_iter,
                    mstep_inner=args.mstep_inner,
                    seed=seed,
                ).fit(samples)
                rows.append(
                    {
                        "method": "gaussian_tpca",
                        "shape": shape,
                        "N": int(N),
                        "rep": int(rep),
                        "seconds": float(time.perf_counter() - start),
                        "n_iter": int(gfit.n_iter),
                        "converged": bool(gfit.converged),
                        "sigma2_hat": float(gfit.sigma2),
                    }
                )

    write_json({"config": vars(args), "rows": rows}, args.output)
