"""Experiment 3 (runtime) rerun with the corrected implementation, 2026-11-17.

Same configurations as the submission (shapes 4x4x4, 6x6x4, 8x8x4; ranks (2,2,2); N in {20, 40};
rho 2; 5 replications; max_iter 20; mstep_inner 1; seeds 300 + 1000 rep + prod(shape) + N).
Methods: L1-TPCA (noise floor kappa 1e-3, rho_init 1, balance) with the mean-only M-step and with the
exact Kronecker second-moment M-step; exact Gaussian TPCA EM.  Records wall-clock seconds and iterations.
"""
import json, time
from pathlib import Path
import numpy as np
import l1_tpca_numpy as T

T.GRAM_ORDER = "c"
# discarded warm-up fit (first call carries interpreter/BLAS start-up cost)
_r = np.random.default_rng(0); _f = T.random_factors((4, 4, 4), (2, 2, 2), _r); _s, _, _ = T.sample_l1_tpca(_f, 2.0, 20, _r)
for _m in (T.L1TPCAVBEM((2, 2, 2), max_iter=3, exact_second_moment=False, tau0_kappa=1e-3), T.GaussianTPCAEM((2, 2, 2), max_iter=3)):
    _m.fit(_s)
rows = []
for shape in [(4, 4, 4), (6, 6, 4), (8, 8, 4)]:
    for N in (20, 40):
        for rep in range(5):
            seed = 300 + 1000 * rep + int(np.prod(shape)) + N
            rng = np.random.default_rng(seed)
            facs = T.random_factors(shape, (2, 2, 2), rng)
            samples, _, _ = T.sample_l1_tpca(facs, 2.0, N, rng)
            for name, model in (
                ("l1_mean_only", T.L1TPCAVBEM((2, 2, 2), rho_init=1.0, max_iter=20, mstep_inner=1, seed=seed, exact_second_moment=False, normalize_mode="balance", tau0_kappa=1e-3)),
                ("l1_exact", T.L1TPCAVBEM((2, 2, 2), rho_init=1.0, max_iter=20, mstep_inner=1, seed=seed, exact_second_moment=True, second_moment_method="kron", normalize_mode="balance", tau0_kappa=1e-3)),
                ("gauss_em", T.GaussianTPCAEM((2, 2, 2), max_iter=20, mstep_inner=1, seed=seed)),
            ):
                t0 = time.perf_counter()
                res = model.fit(samples)
                rows.append({"method": name, "shape": list(shape), "dim": int(np.prod(shape)), "N": N, "rep": rep,
                             "seconds": time.perf_counter() - t0, "n_iter": int(res.n_iter),
                             "subspace_error": T.overall_subspace_error(res.factors, facs)["Err"]})
            print(f"shape={shape} N={N} rep={rep} done", flush=True)
out = Path("../experiments/2026-11-17-exp3-runtime/results"); out.mkdir(parents=True, exist_ok=True)
(out / "exp3_corrected.json").write_text(json.dumps({"rows": rows}, indent=1) + "\n", encoding="utf-8")
print("done", flush=True)
