r"""Verification of the 2026-11-17 numerical fixes (run before any experiment).

1. GIG moments by quadrature vs mpmath (arbitrary precision Bessel) and vs the
   original scipy kve ratios, at p = 100 (synthetic) and p = 16,200 / 23,040 (real data).
2. Kronecker second-moment correction V_k vs the original explicit-matrix code.
3. GaussianTPCAEM: log-likelihood vs a dense multivariate-normal density on a small
   problem, monotone log-likelihood path, and ||x - xhat|| <= ||x - mu|| on every sample.
4. Reproduction of one original Experiment-1 row (legacy settings) to check that the
   revision leaves the synthetic results (Table 2) unchanged up to quadrature error.

Run:  python -u verify_revision.py      (laptop, < 1 min)
"""
import json
import math
import sys

import numpy as np

from l1_tpca_numpy import (
    GaussianTPCAEM,
    L1TPCAVBEM,
    gig_moment,
    gig_moment_bessel_legacy,
    gig_moments,
    gram_kron,
    kron_second_moment_correction,
    multi_mode_dot,
    overall_subspace_error,
    random_factors,
    sample_l1_tpca,
    unfold,
)
from exp_l1_tpca_common import fit_l1_and_gaussian

ok_all = True


def report(name, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


# ---------------------------------------------------------------- 1. GIG
try:
    import mpmath as mp
    mp.mp.dps = 50

    def gig_mp(nu, a, b, s):
        """Independent 50-digit reference.  mpmath.besselk itself fails to converge at
        orders near -p/2 for p in the thousands, so for |nu| > 200 we use tanh-sinh
        quadrature of the u = log(tau) integrand around a mode found by findroot."""
        if abs(nu) <= 200:
            z = mp.sqrt(mp.mpf(a) * b)
            return float((mp.mpf(b) / a) ** (mp.mpf(s) / 2) * mp.besselk(nu + s, z) / mp.besselk(nu, z))
        a_, b_, nu_ = mp.mpf(a), mp.mpf(b), mp.mpf(nu)
        f = lambda u, v: v * u - (a_ * mp.e ** u + b_ * mp.e ** (-u)) / 2
        u0 = mp.findroot(lambda u: nu_ - (a_ * mp.e ** u - b_ * mp.e ** (-u)) / 2, mp.log(b_ / (2 * abs(nu_))))
        f0 = f(u0, nu_)
        w = 1 / mp.sqrt((a_ * mp.e ** u0 + b_ * mp.e ** (-u0)) / 2)
        pts = [u0 + c * w for c in (-60, -20, -6, 0, 6, 20, 60)]
        num = mp.quad(lambda u: mp.e ** (f(u, nu_ + s) - f0), pts)
        den = mp.quad(lambda u: mp.e ** (f(u, nu_) - f0), pts)
        return float(num / den)

    worst = 0.0
    for p in (100, 16200, 23040):
        nu = 1.0 - p / 2.0
        for a in (0.5, 2.0, 362.0):
            for b in (1e-3 * p, 0.3 * p, 1.0 * p, 5.0 * p):
                for s in (1.0, -1.0):
                    ref = gig_mp(nu, a, b, s)
                    got = float(gig_moment(nu, a, b, s))
                    worst = max(worst, abs(got - ref) / abs(ref))
    report("GIG quadrature vs mpmath (p=100,16200,23040; 24 cases x 2 orders)", worst < 1e-8, f"max rel err {worst:.2e}")
    # E[log tau] vs numerical derivative of log K in the order
    nu, a, b = 1.0 - 16200 / 2.0, 2.0, 16200.0
    a_, b_, nu_ = mp.mpf(a), mp.mpf(b), mp.mpf(nu)
    f = lambda u: nu_ * u - (a_ * mp.e ** u + b_ * mp.e ** (-u)) / 2
    u0 = mp.findroot(lambda u: nu_ - (a_ * mp.e ** u - b_ * mp.e ** (-u)) / 2, mp.log(b_ / (2 * abs(nu_))))
    w = 1 / mp.sqrt((a_ * mp.e ** u0 + b_ * mp.e ** (-u0)) / 2)
    pts = [u0 + c * w for c in (-60, -20, -6, 0, 6, 20, 60)]
    f0 = f(u0)
    ref = float(mp.quad(lambda u: u * mp.e ** (f(u) - f0), pts) / mp.quad(lambda u: mp.e ** (f(u) - f0), pts))
    got = float(gig_moments(nu, a, np.array([b]))[2][0])
    report("E[log tau] vs mpmath quadrature at p=16200", abs(got - ref) < 1e-8 * max(1, abs(ref)), f"got {got:.10f} ref {ref:.10f}")
except ImportError:
    report("mpmath available", False, "(install mpmath to run check 1)")

legacy = gig_moment_bessel_legacy(1 - 16200 / 2, 2.0, np.array([16200.0]), -1.0)[0]
report("legacy Bessel code reproduces the overflow at p=16200", legacy == 1e-12, f"legacy E[1/tau] = {legacy}")
new_small = gig_moment(1 - 50.0, 2.0, np.array([100.0]), -1.0)[0]
old_small = gig_moment_bessel_legacy(1 - 50.0, 2.0, np.array([100.0]), -1.0)[0]
report("new == legacy at p=100 (where legacy is valid)", abs(new_small - old_small) / old_small < 1e-9, f"{new_small:.12g} vs {old_small:.12g}")

# ---------------------------------------------------------------- 2. V_k
rng = np.random.default_rng(1)
shape, ranks = (5, 4, 3), (2, 3, 2)
factors = random_factors(shape, ranks, rng)
q = int(np.prod(ranks))
B = rng.normal(size=(q, q))
S = B @ B.T / q
legacy_model = L1TPCAVBEM(ranks, second_moment_method="legacy", max_second_moment_dim=10**9)
Z = rng.normal(size=ranks)
worst = 0.0
for k in range(3):
    Hk = unfold(multi_mode_dot(Z, factors, skip=k), k)
    C_old = legacy_model._second_moment_C(Hk, S, factors, k)
    C_new = Hk @ Hk.T + kron_second_moment_correction(S, factors, ranks, k)
    worst = max(worst, float(np.max(np.abs(C_old - C_new)) / np.max(np.abs(C_old))))
report("Kronecker V_k == explicit V_k (3 modes)", worst < 1e-12, f"max rel diff {worst:.2e}")

# ---------------------------------------------------------------- 3. Gaussian EM
rng = np.random.default_rng(2)
shape, ranks = (4, 3, 3), (2, 2, 1)
true = random_factors(shape, ranks, rng)
L = np.array([[1.0]])
for A in reversed(true):
    L = np.kron(L, A)
X = [multi_mode_dot(rng.normal(size=ranks), true) + 0.7 * rng.normal(size=shape) + 0.3 for _ in range(60)]
g = GaussianTPCAEM(ranks, max_iter=200, tol=1e-10)
fit = g.fit(X)
# dense check of the log-likelihood at the fitted parameters (vec in Fortran order matches L = A_r kron ... kron A_1)
Lf = np.array([[1.0]])
for A in reversed(fit.factors):
    Lf = np.kron(Lf, A)
Sig = Lf @ Lf.T + fit.sigma2 * np.eye(Lf.shape[0])
_, logdet = np.linalg.slogdet(Sig)
Sinv = np.linalg.inv(Sig)
mu = fit.mean.reshape(-1, order="F")
dense = sum(-0.5 * (Sig.shape[0] * math.log(2 * math.pi) + logdet + (x.reshape(-1, order="F") - mu) @ Sinv @ (x.reshape(-1, order="F") - mu)) for x in X)
woodbury = g.loglik(X, fit.mean, fit.factors, fit.sigma2)
report("Gaussian EM: Woodbury loglik == dense MVN loglik", abs(dense - woodbury) < 1e-8 * abs(dense), f"{woodbury:.10f} vs {dense:.10f}")
path = np.array(fit.objective_path)
report("Gaussian EM: loglik nondecreasing", (not g.monotone_violation) and np.all(np.diff(path) >= -1e-9 * np.abs(path[1:])), f"{len(path)-1} iters, converged={fit.converged}")
recon_ok = all(
    np.linalg.norm(x - fit.mean - multi_mode_dot(z, fit.factors)) <= np.linalg.norm(x - fit.mean) + 1e-12
    for x, z in zip(X, fit.z_means)
)
report("Gaussian EM: ||x - xhat|| <= ||x - mu|| for all samples", recon_ok)
err = overall_subspace_error(fit.factors, true)["Err"]
report("Gaussian EM recovers subspace on Gaussian data", err < 0.15, f"subspace error {err:.4f}")

# ---------------------------------------------------------------- 3b. Gram ordering
import l1_tpca_numpy as T
rng = np.random.default_rng(0)
f = random_factors((5, 5, 4), (2, 3, 2), rng)
qq = 12
Lc = np.stack([multi_mode_dot(e.reshape(2, 3, 2), f).reshape(-1) for e in np.eye(qq)], axis=1)
report("gram_kron == L^T L for C-order cores", np.allclose(gram_kron(f), Lc.T @ Lc, rtol=1e-12, atol=1e-10))
T.GRAM_ORDER = "legacy"
legacy_diff = np.abs(gram_kron(f) - Lc.T @ Lc).max() / np.abs(Lc.T @ Lc).max()
report("legacy gram_kron differs from L^T L (documents the original bug)", legacy_diff > 1e-3, f"rel diff {legacy_diff:.3f}")

# ---------------------------------------------------------------- 4. Exp-1 reproduction (legacy ordering, legacy settings)
orig = json.load(open("results_original/exp1_model_specification.json"))
row = [r for r in orig["rows"] if r["N"] == 20 and r["rho_true"] == 2.0 and r["rep"] == 0][0]
seed = 100 + 1000 * 0 + 10 * 20 + int(100 * 2.0)
rng = np.random.default_rng(seed)
facs = random_factors((5, 5, 4), (2, 2, 2), rng)
samples, _, _ = sample_l1_tpca(facs, 2.0, 20, rng)
m = fit_l1_and_gaussian(samples, facs, (2, 2, 2), max_iter=150, mstep_inner=5, seed=seed, rho_init=2.0)
d_l1 = abs(m["l1_tpca"]["subspace_error"]["Err"] - row["l1_tpca"]["subspace_error"]["Err"])
d_g = abs(m["gaussian_tpca"]["subspace_error"]["Err"] - row["gaussian_tpca"]["subspace_error"]["Err"])
report("Exp-1 row (N=20, rho=2, rep 0) reproduced with revised code (legacy settings)", d_l1 < 1e-6 and d_g < 1e-12,
       f"L1 {m['l1_tpca']['subspace_error']['Err']:.8f} vs {row['l1_tpca']['subspace_error']['Err']:.8f}; "
       f"Gauss {m['gaussian_tpca']['subspace_error']['Err']:.8f} vs {row['gaussian_tpca']['subspace_error']['Err']:.8f}")
T.GRAM_ORDER = "c"

# ---------------------------------------------------------------- 5. Noise floor (truncated GIG)
try:
    import mpmath as mp
    mp.mp.dps = 50

    def trunc_ref(nu, a, b, s, lower):
        a_, b_, nu_ = mp.mpf(a), mp.mpf(b), mp.mpf(nu)
        f = lambda u, v: v * u - (a_ * mp.e ** u + b_ * mp.e ** (-u)) / 2
        um = mp.log(lower)
        u0 = mp.findroot(lambda u: nu_ - (a_ * mp.e ** u - b_ * mp.e ** (-u)) / 2, mp.log(b_ / (2 * abs(nu_) + 1)))
        start = max(u0, um)
        f0 = f(start, nu_)
        w = 1 / mp.sqrt((a_ * mp.e ** start + b_ * mp.e ** (-start)) / 2)
        pts = [um] + [start + c * w for c in (0, 6, 20, 60) if start + c * w > um]
        num = mp.quad(lambda u: mp.e ** (f(u, nu_ + s) - f0), pts)
        den = mp.quad(lambda u: mp.e ** (f(u, nu_) - f0), pts)
        return float(num / den)

    worst = 0.0
    for p in (100, 16200):
        nu = 1.0 - p / 2.0
        for a, b in ((2.0, 0.3 * p), (2.0, 1.0 * p), (0.5, 5.0 * p)):
            mode = float(b / (math.sqrt(nu * nu + a * b) - nu))
            for lower in (0.1 * mode, 0.9 * mode, 3.0 * mode):  # floor below, near and above the mode
                for s in (1.0, -1.0):
                    ref = trunc_ref(nu, a, b, s, lower)
                    got = float(gig_moment(nu, a, b, s, lower=lower))
                    worst = max(worst, abs(got - ref) / abs(ref))
    report("truncated GIG (noise floor) vs mpmath quadrature (36 cases)", worst < 1e-8, f"max rel err {worst:.2e}")
except ImportError:
    report("mpmath available", False)

rng = np.random.default_rng(320)
facs = random_factors((5, 5, 4), (2, 2, 2), rng)
samples, _, _ = sample_l1_tpca(facs, 2.0, 20, rng)
free = L1TPCAVBEM((2, 2, 2), rho_init=1.0, max_iter=150, mstep_inner=5, seed=320, normalize_mode="balance").fit(samples)
floored = L1TPCAVBEM((2, 2, 2), rho_init=1.0, max_iter=150, mstep_inner=5, seed=320, normalize_mode="balance", tau0_kappa=1e-3).fit(samples)
report("without floor the singular weight appears (documents R2-M1)", free.weights.max() > 1e6, f"max weight {free.weights.max():.3g}")
report("with floor all weights <= 1/tau0", floored.weights.max() <= 1.0 / floored.tau0 * (1 + 1e-9),
       f"tau0 {floored.tau0:.4g}, max weight {floored.weights.max():.4g} <= {1 / floored.tau0:.4g}")

print("ALL PASS" if ok_all else "SOME CHECKS FAILED", flush=True)
sys.exit(0 if ok_all else 1)
