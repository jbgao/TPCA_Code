from __future__ import annotations

import argparse
import json
import numpy as np
import torch

from riem_tpca_v4 import (
    GeometricTPCAPower,
    RiemannianGaussianTPCAV4,
    compare_factor_subspaces,
    random_tpca_factors,
    sample_tpca,
    resolve_device,
)


def parse_int_tuple(s: str):
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo for anchored Riemannian TPCA refinement (v4)")
    parser.add_argument("--shape", type=str, default="5,5,5")
    parser.add_argument("--ranks", type=str, default="2,2,2")
    parser.add_argument("--N", type=int, default=50)
    parser.add_argument("--sigma2", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1523)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--power_iter", type=int, default=20)
    parser.add_argument("--riem_iter", type=int, default=500)
    parser.add_argument("--power_tol", type=float, default=1e-6)
    parser.add_argument("--riem_tol", type=float, default=1e-7)
    parser.add_argument("--lr_u", type=float, default=0.01)
    parser.add_argument("--lr_s", type=float, default=0.001)
    parser.add_argument("--lr_gamma", type=float, default=0.001)
    parser.add_argument("--lr_sigma", type=float, default=0.001)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--freeze_sigma_iters", type=int, default=20)
    parser.add_argument("--freeze_factor_iters", type=int, default=100)
    parser.add_argument("--anchor_lambda", type=float, default=50.0)
    parser.add_argument("--anchor_decay", type=float, default=50.0)
    parser.add_argument("--no_power_init", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    args.no_power_init = False
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

    init_factors = None if args.no_power_init else power_res.factors

    riem = RiemannianGaussianTPCAV4(
        shape=shape,
        ranks=ranks,
        lr_u=args.lr_u,
        lr_s=args.lr_s,
        lr_gamma=args.lr_gamma,
        lr_sigma=args.lr_sigma,
        grad_clip=args.grad_clip,
        freeze_sigma_iters=args.freeze_sigma_iters,
        freeze_factor_iters=args.freeze_factor_iters,
        sigma2_init=args.sigma2,
        gamma_init=1.0,
        anchor_lambda=args.anchor_lambda,
        anchor_decay=args.anchor_decay,
        max_iter=args.riem_iter,
        tol=args.riem_tol,
        device=device,
        dtype=dtype,
        verbose=args.verbose,
    )
    riem_res = riem.fit(samples, rng=rng, init_factors=init_factors)

    power_errs = compare_factor_subspaces(true_factors, power_res.factors)
    riem_errs = compare_factor_subspaces(true_factors, riem_res.factors)

    summary = {
        "shape": shape,
        "ranks": ranks,
        "N": args.N,
        "sigma2_true": args.sigma2,
        "device": str(device),
        "power_init_used": not args.no_power_init,
        "power": {
            "subspace_errors": power_errs,
            "objective_last": power_res.history["objective"][-1] if power_res.history["objective"] else None,
            "iters": len(power_res.history["objective"]),
        },
        "riemannian_v4": {
            "subspace_errors": riem_errs,
            "sigma2_est": riem_res.sigma2,
            "gamma_est": riem_res.gamma,
            "loss_last": riem_res.history["loss"][-1] if riem_res.history["loss"] else None,
            "anchor_last": riem_res.history["anchor"][-1] if riem_res.history.get("anchor") else None,
            "iters": len(riem_res.history["loss"]),
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
