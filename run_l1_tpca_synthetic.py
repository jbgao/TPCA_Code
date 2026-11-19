from __future__ import annotations

import argparse
import json

import numpy as np

from l1_tpca_numpy import (
    L1TPCAVBEM,
    contaminate_tensor_samples,
    overall_subspace_error,
    random_factors,
    sample_l1_tpca,
)


def parse_shape(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(","))


def summarize_weights(weights: np.ndarray, outlier_mask: np.ndarray) -> dict[str, float]:
    clean = weights[~outlier_mask]
    out = weights[outlier_mask]
    return {
        "clean_mean": float(clean.mean()) if clean.size else float("nan"),
        "outlier_mean": float(out.mean()) if out.size else float("nan"),
        "clean_median": float(np.median(clean)) if clean.size else float("nan"),
        "outlier_median": float(np.median(out)) if out.size else float("nan"),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Synthetic example for Algorithm 1 L1-TPCA VBEM.")
    ap.add_argument("--shape", type=parse_shape, default="4,5,3")
    ap.add_argument("--ranks", type=parse_shape, default="2,2,1")
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--rho", type=float, default=2.0, help="True Exp(rate=rho/2) precision parameter.")
    ap.add_argument("--rho_init", type=float, default=1.0)
    ap.add_argument("--contamination_rate", type=float, default=0.15)
    ap.add_argument("--outlier_scale", type=float, default=8.0)
    ap.add_argument("--max_iter", type=int, default=25)
    ap.add_argument("--mstep_inner", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    true_factors = random_factors(args.shape, args.ranks, rng)
    clean_samples, _, true_taus = sample_l1_tpca(true_factors, args.rho, args.N, rng)
    samples, outlier_mask = contaminate_tensor_samples(
        clean_samples,
        rng,
        contamination_rate=args.contamination_rate,
        outlier_scale=args.outlier_scale,
    )

    model = L1TPCAVBEM(
        args.ranks,
        rho_init=args.rho_init,
        gamma_a=2.0,
        gamma_b=0.5,
        update_rho=True,
        max_iter=args.max_iter,
        tol=1e-4,
        mstep_inner=args.mstep_inner,
        init="hosvd",
        verbose=args.verbose,
        seed=args.seed,
    )
    fit = model.fit(samples)
    subspace = overall_subspace_error(fit.factors, true_factors)

    payload = {
        "config": vars(args),
        "true_tau_summary": {
            "mean": float(true_taus.mean()),
            "median": float(np.median(true_taus)),
        },
        "fit": {
            "rho_hat": float(fit.rho),
            "n_iter": int(fit.n_iter),
            "converged": bool(fit.converged),
            "objective_first": float(fit.objective_path[0]) if fit.objective_path else None,
            "objective_last": float(fit.objective_path[-1]) if fit.objective_path else None,
            "subspace_error": subspace,
            "weight_summary": summarize_weights(fit.weights, outlier_mask),
        },
        "outliers": {
            "count": int(outlier_mask.sum()),
            "indices": np.flatnonzero(outlier_mask).astype(int).tolist(),
        },
    }
    print(json.dumps(payload, indent=2))
