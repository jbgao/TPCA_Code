r"""ELBO / log-likelihood traces for the revision (R2 Major Comment 2), pre-registered 2026-11-17.

Reruns, with the final corrected algorithm (noise floor kappa 1e-3, rho_init 1, balance normalisation,
exact Kronecker M-step), exactly the data/seeds of
  * Experiment 1 (all 30 runs: N in {20,40}, rho in {0.75,2,6}, 5 replications; max_iter 150, mstep_inner 5), and
  * Experiment 2, whole-tensor contamination at rate 0.2 and outlier scale 12 (10 replications; max_iter 40,
    mstep_inner 2),
and stores the full objective path of every fit: the L1-TPCA ELBO and the exact Gaussian TPCA log-likelihood.
Run (laptop, a few minutes):  python -u exp_elbo_traces.py
"""
import json
import time
from pathlib import Path

import numpy as np

import l1_tpca_numpy as T

T.GRAM_ORDER = "c"
OUT = Path("../experiments/2026-11-17-elbo-traces/results")
OUT.mkdir(parents=True, exist_ok=True)


def fits(samples, ranks, max_iter, mstep_inner, seed):
    l1 = T.L1TPCAVBEM(ranks, rho_init=1.0, max_iter=max_iter, mstep_inner=mstep_inner, seed=seed,
                      exact_second_moment=True, normalize_mode="balance", second_moment_method="kron",
                      tau0_kappa=1e-3).fit(samples)
    gm = T.GaussianTPCAEM(ranks, max_iter=max_iter, mstep_inner=mstep_inner, seed=seed)
    g = gm.fit(samples)
    return {"l1_elbo": [float(v) for v in l1.objective_path], "l1_converged": bool(l1.converged),
            "gauss_loglik": [float(v) for v in g.objective_path], "gauss_converged": bool(g.converged),
            "gauss_monotone_violation": bool(gm.monotone_violation)}


rows = []
t0 = time.time()
for N in (20, 40):
    for rho in (0.75, 2.0, 6.0):
        for rep in range(5):
            seed = 100 + 1000 * rep + 10 * N + int(100 * rho)
            rng = np.random.default_rng(seed)
            facs = T.random_factors((5, 5, 4), (2, 2, 2), rng)
            samples, _, _ = T.sample_l1_tpca(facs, rho, N, rng)
            rows.append({"exp": 1, "N": N, "rho": rho, "rep": rep, **fits(samples, (2, 2, 2), 150, 5, seed)})
            print(f"exp1 N={N} rho={rho} rep={rep} ({time.time() - t0:.0f}s)", flush=True)
for rep in range(10):
    rate, scale = 0.2, 12.0
    seed = 200 + rep * 10000 + int(1000 * rate) + int(10 * scale)
    rng = np.random.default_rng(seed)
    facs = T.random_factors((5, 5, 4), (2, 2, 2), rng)
    clean, _, _ = T.sample_l1_tpca(facs, 2.0, 50, rng)
    samples, _ = T.contaminate_tensor_samples(clean, rng, contamination_rate=rate, outlier_scale=scale)
    rows.append({"exp": 2, "type": "tensor", "rate": rate, "scale": scale, "rep": rep, **fits(samples, (2, 2, 2), 40, 2, seed)})
    print(f"exp2 rep={rep} ({time.time() - t0:.0f}s)", flush=True)
(OUT / "elbo_traces.json").write_text(json.dumps({"rows": rows}) + "\n", encoding="utf-8")
print("done", flush=True)
