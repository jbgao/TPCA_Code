from __future__ import annotations

import argparse
import json
import numpy as np
import torch

from riem_tpca import (
    GeometricTPCAPower,
    RiemannianGaussianTPCA,
    compare_factor_subspaces,
    random_tpca_factors,
    sample_tpca,
    resolve_device,
)


def parse_int_tuple(s: str):
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo for geometric / Riemannian TPCA project")
    parser.add_argument("--shape", type=str, default="5,5,5")
    parser.add_argument("--ranks", type=str, default="2,2,2")
    parser.add_argument("--N", type=int, default=40)
    parser.add_argument("--sigma2", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--power_iter", type=int, default=20)
    parser.add_argument("--riem_iter", type=int, default=120)
    parser.add_argument("--power_tol", type=float, default=1e-6)
    parser.add_argument("--riem_tol", type=float, default=1e-7)
    parser.add_argument("--lr_u", type=float, default=0.2)
    parser.add_argument("--lr_s", type=float, default=0.05)
    parser.add_argument("--lr_sigma", type=float, default=0.05)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    shape = parse_int_tuple(args.shape)
    ranks = parse_int_tuple(args.ranks)
    if len(shape) != len(ranks):
        raise ValueError("shape and ranks must have the same length")

    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)
    dtype = torch.float64

    true_factors = random_tpca_factors(shape, ranks, rng=rng, device=device, dtype=dtype, spectrum="linear")
    samples = sample_tpca(true_factors, args.sigma2, args.N, rng=rng, device=device, dtype=dtype)

    power = GeometricTPCAPower(
        shape=shape,
        ranks=ranks,
        max_iter=args.power_iter,
        tol=args.power_tol,
        device=device,
        dtype=dtype,
        verbose=args.verbose,
    )
    power_res = power.fit(samples, rng=rng)

    riem = RiemannianGaussianTPCA(
        shape=shape,
        ranks=ranks,
        lr_u=args.lr_u,
        lr_s=args.lr_s,
        lr_sigma=args.lr_sigma,
        max_iter=args.riem_iter,
        tol=args.riem_tol,
        device=device,
        dtype=dtype,
        verbose=args.verbose,
    )
    riem_res = riem.fit(samples, rng=rng)

    power_errs = compare_factor_subspaces(true_factors, power_res.factors)
    riem_errs = compare_factor_subspaces(true_factors, riem_res.factors)

    summary = {
        "shape": shape,
        "ranks": ranks,
        "N": args.N,
        "sigma2_true": args.sigma2,
        "device": str(device),
        "power": {
            "subspace_errors": power_errs,
            "objective_last": power_res.history["objective"][-1] if power_res.history["objective"] else None,
            "iters": len(power_res.history["objective"]),
        },
        "riemannian": {
            "subspace_errors": riem_errs,
            "sigma2_est": riem_res.sigma2,
            "loss_last": riem_res.history["loss"][-1] if riem_res.history["loss"] else None,
            "iters": len(riem_res.history["loss"]),
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
