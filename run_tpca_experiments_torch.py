from __future__ import annotations

import argparse
import json
import numpy as np

from tpca_reimplementation_torch import case1_em_once, case2_power_once, case3_power_once


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the three numerical settings from the TPCA paper with a PyTorch backend.")
    parser.add_argument("--case", choices=["case1", "case2", "case3"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replications", type=int, default=1)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--sigma2", type=float, default=1.0)
    parser.add_argument("--N", type=int, default=15)
    parser.add_argument("--L", type=int, default=5, help="number of power iterations for case2/case3")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, ...")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out = []
    if args.case == "case1":
        for _ in range(args.replications):
            out.append(case1_em_once(args.n, args.m, args.sigma2, rng, n_samples=args.N, device=args.device))
    elif args.case == "case2":
        for _ in range(args.replications):
            out.append(case2_power_once(rng, n_iter=args.L, device=args.device))
    else:
        shape = (40, 60, 80)
        ranks = (2, 3, 4)
        for _ in range(args.replications):
            out.append(case3_power_once(shape, ranks, args.sigma2, args.N, rng, n_iter=args.L, device=args.device))

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
