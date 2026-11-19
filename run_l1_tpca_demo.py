from __future__ import annotations

import argparse
import json
import numpy as np

from l1_tpca_torch import (
    random_factors,
    sample_l1_tpca,
    L1TPCAVBEM,
    L1TPCARobustPower,
    overall_factor_error,
)
from tpca_reimplementation_torch import TPCAEM


def parse_shape(s: str):
    return tuple(int(x) for x in s.split(","))


def main() -> None:
    ap = argparse.ArgumentParser(description="Starter experiments for the L1-TPCA project.")
    ap.add_argument("--shape", type=parse_shape, default="6,6,6", help="Tensor shape, e.g. 6,6,6")
    ap.add_argument("--ranks", type=parse_shape, default="2,2,2", help="Tucker ranks, e.g. 2,2,2")
    ap.add_argument("--lam", type=float, default=2.0, help="Latent-scale exponential rate parameter (Exp(rate=lam/2)).")
    ap.add_argument("--N", type=int, default=50, help="Number of tensor samples.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vbem_iter", type=int, default=100)
    ap.add_argument("--power_iter", type=int, default=20)
    ap.add_argument("--gaussian_iter", type=int, default=100)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    true_factors = random_factors(args.shape, args.ranks, rng, device=args.device)
    samples, taus = sample_l1_tpca(true_factors, lam=args.lam, n_samples=args.N, rng=rng, device=args.device)

    robust_vbem = L1TPCAVBEM(
        ranks=args.ranks,
        lam_init=args.lam,
        max_iter=args.vbem_iter,
        verbose=False,
        device=args.device,
    )
    fit_vbem = robust_vbem.fit(samples)
    err_vbem = overall_factor_error(fit_vbem.factors, true_factors)

    robust_power = L1TPCARobustPower(
        ranks=args.ranks,
        lam=args.lam,
        n_iter=args.power_iter,
        device=args.device,
    )
    fit_power = robust_power.fit(samples)
    err_power = overall_factor_error(fit_power.factors, true_factors)

    # Gaussian baseline: initialize sigma^2 at E[tau] = 2/lam.
    sigma2_init = 2.0 / args.lam
    gaussian = TPCAEM(
        ranks=args.ranks,
        sigma2_init=sigma2_init,
        max_iter=args.gaussian_iter,
        tol=1e-4,
        init="hosvd",
        update_nu=False,
        center=True,
        verbose=False,
        device=args.device,
    )
    fit_gauss = gaussian.fit(samples)
    err_gauss = overall_factor_error(fit_gauss.factors, true_factors)

    result = {
        "shape": tuple(args.shape),
        "ranks": tuple(args.ranks),
        "N": args.N,
        "lam_true": args.lam,
        "tau_mean_true": float(np.mean(taus)),
        "robust_vbem": {
            "Err": err_vbem["Err"],
            "mode_errors": err_vbem["mode_errors"],
            "lam_hat": fit_vbem.lam,
            "n_iter": fit_vbem.n_iter,
            "converged": fit_vbem.converged,
            "device": fit_vbem.device,
        },
        "robust_power": {
            "Err": err_power["Err"],
            "mode_errors": err_power["mode_errors"],
            "n_iter": fit_power.n_iter,
            "device": fit_power.device,
        },
        "gaussian_em_baseline": {
            "Err": err_gauss["Err"],
            "mode_errors": err_gauss["mode_errors"],
            "sigma2_hat": fit_gauss.sigma2,
            "n_iter": fit_gauss.n_iter,
            "converged": fit_gauss.converged,
            "device": fit_gauss.device,
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
