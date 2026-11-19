from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple
import math

import numpy as np


Array = np.ndarray


def unfold(X: Array, mode: int) -> Array:
    X = np.asarray(X)
    order = (mode,) + tuple(i for i in range(X.ndim) if i != mode)
    return np.transpose(X, order).reshape(X.shape[mode], -1)


def fold(Xk: Array, shape: Sequence[int], mode: int) -> Array:
    shape = tuple(shape)
    order = (mode,) + tuple(i for i in range(len(shape)) if i != mode)
    inv_order = np.argsort(order)
    full = Xk.reshape([shape[mode]] + [shape[i] for i in range(len(shape)) if i != mode])
    return np.transpose(full, inv_order)


def mode_dot(X: Array, A: Array, mode: int) -> Array:
    X = np.asarray(X)
    A = np.asarray(A)
    Yk = A @ unfold(X, mode)
    shape = list(X.shape)
    shape[mode] = A.shape[0]
    return fold(Yk, shape, mode)


def multi_mode_dot(
    X: Array,
    factors: Sequence[Array],
    *,
    skip: Optional[int] = None,
    transpose: bool = False,
) -> Array:
    Y = np.asarray(X)
    for k, A in enumerate(factors):
        if skip is not None and k == skip:
            continue
        M = A.T if transpose else A
        Y = mode_dot(Y, M, k)
    return Y


# Ordering of the Kronecker Gram matrix (revision 2026-11-17).
# Cores are vectorised in C order (Z.reshape(-1)), for which
#     vec_C(Z x_1 A_1 ... x_r A_r) = (A_1 kron ... kron A_r) vec_C(Z),
# so L^T L = G_1 kron ... kron G_r.  The original code built G_r kron ... kron G_1,
# i.e. a mode-reversed (wrong) posterior precision.  "c" is correct; "legacy"
# reproduces the original runs (Experiments 1-5 of the submission) bit for bit.
GRAM_ORDER = "c"


def gram_kron(factors: Sequence[Array]) -> Array:
    """L^T L for C-order core vectorisation: A_1^T A_1 kron ... kron A_r^T A_r.
    With GRAM_ORDER == "legacy": the original (mode-reversed) product."""
    seq = list(reversed(factors)) if GRAM_ORDER == "legacy" else list(factors)
    out = np.array([[1.0]])
    for A in seq:
        A = np.asarray(A, dtype=float)
        out = np.kron(out, A.T @ A)
    return out


def _check_samples(samples: Sequence[Array]) -> Tuple[List[Array], Tuple[int, ...]]:
    if len(samples) == 0:
        raise ValueError("samples must be non-empty")
    arrs = [np.asarray(X, dtype=float) for X in samples]
    shape = tuple(arrs[0].shape)
    if any(tuple(X.shape) != shape for X in arrs):
        raise ValueError("all samples must have the same shape")
    return arrs, shape


def center_samples(samples: Sequence[Array]) -> Tuple[List[Array], Array]:
    mean = np.mean(np.stack(samples, axis=0), axis=0)
    return [np.asarray(X, dtype=float) - mean for X in samples], mean


def random_factors(
    shape: Sequence[int],
    ranks: Sequence[int],
    rng: np.random.Generator,
    *,
    scale_like_paper: bool = True,
) -> List[Array]:
    factors: List[Array] = []
    for n, m in zip(shape, ranks):
        Araw = rng.normal(size=(n, m))
        U, s, _ = np.linalg.svd(Araw, full_matrices=False)
        A = U[:, :m] @ np.diag(s[:m])
        if scale_like_paper:
            target = math.sqrt(n * m)
            A = A * (target / max(np.linalg.norm(A, ord="fro"), 1e-12))
        factors.append(A)
    return factors


def hosvd_warm_start(
    samples: Sequence[Array],
    ranks: Sequence[int],
    *,
    scale_to_sqrt_n: bool = True,
) -> List[Array]:
    samples, shape = _check_samples(samples)
    factors: List[Array] = []
    for k, (n, m) in enumerate(zip(shape, ranks)):
        stacked = np.concatenate([unfold(X, k) for X in samples], axis=1)
        U, _, _ = np.linalg.svd(stacked, full_matrices=False)
        A = U[:, :m]
        if scale_to_sqrt_n:
            A = A * math.sqrt(n)
        factors.append(A)
    return factors


def sample_l1_tpca(
    factors: Sequence[Array],
    rho: float,
    n_samples: int,
    rng: np.random.Generator,
    *,
    mean: Optional[Array] = None,
) -> Tuple[List[Array], Array, Array]:
    """Draw samples from the paper's Laplace scale-mixture TPCA model.

    vec(E_i) | tau_i ~ N(0, tau_i I), tau_i ~ Exp(rate=rho/2).
    """
    shape = tuple(A.shape[0] for A in factors)
    ranks = tuple(A.shape[1] for A in factors)
    mean_tensor = np.zeros(shape) if mean is None else np.asarray(mean, dtype=float)
    samples: List[Array] = []
    z_true: List[Array] = []
    taus: List[float] = []
    for _ in range(n_samples):
        Z = rng.normal(size=ranks)
        tau = float(rng.exponential(scale=2.0 / max(rho, 1e-12)))
        noise = rng.normal(scale=math.sqrt(tau), size=shape)
        samples.append(mean_tensor + multi_mode_dot(Z, factors) + noise)
        z_true.append(Z)
        taus.append(tau)
    return samples, np.asarray(z_true), np.asarray(taus)


def contaminate_tensor_samples(
    samples: Sequence[Array],
    rng: np.random.Generator,
    *,
    contamination_rate: float = 0.1,
    outlier_scale: float = 8.0,
) -> Tuple[List[Array], Array]:
    out = [np.asarray(X, dtype=float).copy() for X in samples]
    n_out = max(1, int(round(contamination_rate * len(out)))) if contamination_rate > 0 else 0
    mask = np.zeros(len(out), dtype=bool)
    if n_out == 0:
        return out, mask
    idx = rng.choice(len(out), size=n_out, replace=False)
    for i in idx:
        mask[i] = True
        out[i] = out[i] + rng.normal(scale=outlier_scale, size=out[i].shape)
    return out, mask


def _k_ratio(nu_num: float, nu_den: float, z: Array) -> Array:
    z = np.maximum(np.asarray(z, dtype=float), 1e-12)
    try:
        from scipy.special import kve  # type: ignore

        return kve(nu_num, z) / np.maximum(kve(nu_den, z), 1e-300)
    except Exception:
        import mpmath as mp

        vals = []
        for zz in z.reshape(-1):
            num = mp.besselk(nu_num, float(zz))
            den = mp.besselk(nu_den, float(zz))
            vals.append(float(num / max(den, mp.mpf("1e-300"))))
        return np.asarray(vals, dtype=float).reshape(z.shape)


def _gammaln(x: float) -> float:
    try:
        from scipy.special import gammaln  # type: ignore

        return float(gammaln(x))
    except Exception:
        return float(math.lgamma(x))


def _digamma(x: float) -> float:
    try:
        from scipy.special import digamma  # type: ignore

        return float(digamma(x))
    except Exception:
        import mpmath as mp

        return float(mp.digamma(x))


def _log_kv(nu: float, z: Array) -> Array:
    z = np.maximum(np.asarray(z, dtype=float), 1e-12)
    try:
        from scipy.special import kve  # type: ignore

        return np.log(np.maximum(kve(nu, z), 1e-300)) - z
    except Exception:
        import mpmath as mp

        vals = []
        for zz in z.reshape(-1):
            vals.append(float(mp.log(mp.besselk(nu, float(zz)))))
        return np.asarray(vals, dtype=float).reshape(z.shape)


# ---------------------------------------------------------------------------
# GIG moments by quadrature in u = log(tau)   (revision 2026-11-17)
#
# The original implementation evaluated ratios of exponentially scaled Bessel
# functions K_nu with nu = 1 - p/2.  For real-data tensors (p >= 16,200) these
# overflow in double precision and the moments were silently clipped to 1e-12,
# which disabled the robust weighting.  Here every GIG quantity is computed from
#     Z(nu, a, b) = int_0^inf tau^(nu-1) exp(-(a tau + b/tau)/2) dtau
#                 = int_R exp(f(u)) du,   f(u) = nu u - (a e^u + b e^-u)/2,
# which is strictly concave in u.  We locate the mode in closed form, bracket the
# region where f >= f(mode) - GIG_DROP by bisection, and integrate with the
# trapezoid rule in log-sum-exp form.  No Bessel function is formed, and
# non-finite results raise instead of being clipped.
# ---------------------------------------------------------------------------
GIG_GRID = 801
GIG_DROP = 60.0


def _gig_f(u: Array, nu: float, a: float, b: Array) -> Array:
    return nu * u - 0.5 * (a * np.exp(u) + b * np.exp(-u))


def _gig_grid(nu: float, a: float, b: Array, lower: float = 0.0) -> Tuple[Array, Array]:
    """Return (u_grid, log_integrand) with shapes (n_b, GIG_GRID).

    lower > 0 truncates the GIG to tau >= lower (noise floor, revision 2026-11-17):
    the integrand is restricted to u >= log(lower), and the support is bracketed from
    max(mode, log lower), where the truncated log-integrand attains its maximum."""
    b = np.atleast_1d(np.asarray(b, dtype=float))
    a = float(a)
    if not (a > 0 and np.all(b > 0)):
        raise ValueError("GIG requires a > 0 and b > 0")
    root = np.sqrt(nu * nu + a * b)
    if nu >= 0:
        e_mode = (nu + root) / a
    else:
        e_mode = b / (root - nu)  # stable form of (nu + root)/a for nu < 0
    u_mode = np.log(e_mode)
    f_mode = _gig_f(u_mode, nu, a, b)
    curv = 0.5 * (a * e_mode + b / e_mode)  # -f''(mode) > 0
    scale = 1.0 / np.sqrt(curv)
    u_min = math.log(lower) if lower > 0 else -np.inf
    start = np.maximum(u_mode, u_min)
    target = _gig_f(start, nu, a, b) - GIG_DROP

    def bracket(sign: float) -> Array:
        step = scale.copy()
        for _ in range(200):  # expand until f drops below target
            u = start + sign * step
            bad = _gig_f(u, nu, a, b) > target
            if not np.any(bad):
                break
            step = np.where(bad, 2.0 * step, step)
        lo = np.zeros_like(step)
        hi = step
        for _ in range(60):  # bisection on the distance from the mode
            mid = 0.5 * (lo + hi)
            above = _gig_f(start + sign * mid, nu, a, b) > target
            lo = np.where(above, mid, lo)
            hi = np.where(above, hi, mid)
        return start + sign * hi

    with np.errstate(over="ignore"):
        u_lo, u_hi = np.maximum(bracket(-1.0), u_min), bracket(1.0)
    t = np.linspace(0.0, 1.0, GIG_GRID)
    u = u_lo[:, None] + (u_hi - u_lo)[:, None] * t[None, :]
    return u, _gig_f(u, nu, a, b[:, None])


def _quad_weights(n: int) -> Array:
    """Composite Simpson weights (n odd), relative to the step h; accurate also when the
    noise floor truncates the integrand at a non-zero value."""
    if n % 2 == 0:
        raise ValueError("GIG_GRID must be odd for Simpson's rule")
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w / 3.0


def _trapz_logsumexp(u: Array, logf: Array) -> Array:
    """log of the integral of exp(logf) over each row of u (Simpson's rule, log-sum-exp)."""
    du = (u[:, -1] - u[:, 0]) / (u.shape[1] - 1)
    w = _quad_weights(u.shape[1])
    m = logf.max(axis=1, keepdims=True)
    return m[:, 0] + np.log(np.sum(w[None, :] * np.exp(logf - m), axis=1) * du)


def gig_log_partition(nu: float, a: float, b: Array | float, lower: float = 0.0) -> Array:
    """log Z(nu, a, b) = log int tau^(nu-1) exp(-(a tau + b/tau)/2) dtau
    = log 2 + (nu/2) log(b/a) + log K_nu(sqrt(ab))   (integral over tau >= lower)."""
    u, logf = _gig_grid(nu, a, b, lower)
    out = _trapz_logsumexp(u, logf)
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("non-finite GIG log-partition")
    return out


def gig_moments(nu: float, a: float, b: Array | float, lower: float = 0.0) -> Tuple[Array, Array, Array]:
    """(E[tau], E[1/tau], E[log tau]) for GIG(nu, a, b) truncated to tau >= lower, on one grid."""
    u, logf = _gig_grid(nu, a, b, lower)
    logz = _trapz_logsumexp(u, logf)
    e_tau = np.exp(_trapz_logsumexp(u, logf + u) - logz)
    e_inv = np.exp(_trapz_logsumexp(u, logf - u) - logz)
    m = logf.max(axis=1, keepdims=True)
    w = np.exp(logf - m) * _quad_weights(u.shape[1])[None, :]
    e_log = np.sum(w * u, axis=1) / np.sum(w, axis=1)
    for name, arr in (("E[tau]", e_tau), ("E[1/tau]", e_inv), ("E[log tau]", e_log)):
        if not np.all(np.isfinite(arr)):
            raise FloatingPointError(f"non-finite GIG moment {name}")
    return e_tau, e_inv, e_log


def gig_moment(nu: float, a: float, b: Array | float, order: float, lower: float = 0.0) -> Array:
    """E[X^order] for GIG(nu, a, b): x^(nu-1) exp(-(a*x + b/x)/2).

    Quadrature-based (see block comment above); valid for any order and for the
    very large |nu| = p/2 - 1 that arise for tensor observations.
    """
    b_arr = np.maximum(np.asarray(b, dtype=float), 1e-300)
    a = max(float(a), 1e-300)
    u, logf = _gig_grid(nu, a, b_arr.reshape(-1), lower)
    logz = _trapz_logsumexp(u, logf)
    out = np.exp(_trapz_logsumexp(u, logf + float(order) * u) - logz)
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("non-finite GIG moment")
    return out.reshape(b_arr.shape)


def gig_expected_log(nu: float, a: float, b: Array | float, lower: float = 0.0) -> Array:
    """E[log X] for GIG(nu, a, b) truncated to tau >= lower, by the same quadrature."""
    b_arr = np.maximum(np.asarray(b, dtype=float), 1e-300)
    _, _, e_log = gig_moments(nu, max(float(a), 1e-300), b_arr.reshape(-1), lower)
    return e_log.reshape(b_arr.shape)


def gig_moment_bessel_legacy(nu: float, a: float, b: Array | float, order: float) -> Array:
    """The ORIGINAL (pre-revision) Bessel-ratio implementation, kept only to
    document and test the overflow.  Do not use for fitting."""
    b_arr = np.maximum(np.asarray(b, dtype=float), 1e-12)
    a = max(float(a), 1e-12)
    z = np.sqrt(a * b_arr)
    moment = (b_arr / a) ** (0.5 * order) * _k_ratio(nu + order, nu, z)
    return np.nan_to_num(moment, nan=1e-12, posinf=1e12, neginf=1e-12)


def kron_second_moment_correction(S: Array, factors: Sequence[Array], ranks: Sequence[int], k: int) -> Array:
    """V_k = E[H_(k) H_(k)^T] - Hbar_(k) Hbar_(k)^T for H = Z x_{s!=k} A_s.

    With vec(Z) in C order and covariance S, unfold(H, k) = Z_(k) (A_s1 kron A_s2 ...)^T
    for the remaining modes s1 < s2 < ..., so V_k[a,b] = sum_{c,d} Cov(Z_(k)[a,c], Z_(k)[b,d]) M[c,d]
    with M = G_s1 kron G_s2 kron ..., G_s = A_s^T A_s.  Cost O(q^2), no size threshold.
    """
    ranks = tuple(int(x) for x in ranks)
    r = len(ranks)
    others = [s for s in range(r) if s != k]
    M = np.array([[1.0]])
    for s in others:
        A = np.asarray(factors[s], dtype=float)
        M = np.kron(M, A.T @ A)
    S4 = np.asarray(S, dtype=float).reshape(ranks + ranks)
    perm = [k] + others + [r + k] + [r + s for s in others]
    rest = int(np.prod([ranks[s] for s in others])) if others else 1
    cov = np.transpose(S4, perm).reshape(ranks[k], rest, ranks[k], rest)
    return np.einsum("acbd,cd->ab", cov, M)


def normalize_factor_list(factors: Sequence[Array], mode: str) -> List[Array]:
    """'fixed'   : original behaviour, ||A_k||_F = sqrt(n_k m_k) for each k (changes L);
    'balance' : equalise ||A_k||_F across modes keeping prod_k ||A_k||_F (L unchanged);
    'none'    : no normalisation."""
    factors = [np.asarray(A, dtype=float) for A in factors]
    if mode == "none":
        return factors
    if mode == "fixed":
        return [A * (math.sqrt(A.shape[0] * A.shape[1]) / max(np.linalg.norm(A, ord="fro"), 1e-12)) for A in factors]
    if mode == "balance":
        norms = np.array([max(np.linalg.norm(A, ord="fro"), 1e-300) for A in factors])
        g = float(np.exp(np.mean(np.log(norms))))
        return [A * (g / n) for A, n in zip(factors, norms)]
    raise ValueError(f"unknown normalisation mode {mode!r}")


def _stable_pinv(A: Array, ridge: float = 1e-8) -> Array:
    A = np.nan_to_num(np.asarray(A, dtype=float), nan=0.0, posinf=1e12, neginf=-1e12)
    try:
        return np.linalg.pinv(A)
    except np.linalg.LinAlgError:
        A = A + ridge * np.eye(A.shape[0], A.shape[1])
        return np.linalg.pinv(A, hermitian=A.shape[0] == A.shape[1])


def subspace_error(A_hat: Array, A_true: Array) -> float:
    Qh, _ = np.linalg.qr(np.asarray(A_hat, dtype=float))
    Qt, _ = np.linalg.qr(np.asarray(A_true, dtype=float))
    Ph = Qh @ Qh.T
    Pt = Qt @ Qt.T
    rank = max(1, A_true.shape[1])
    return float(np.linalg.norm(Ph - Pt, ord="fro") / math.sqrt(2.0 * rank))


def overall_subspace_error(factors_hat: Sequence[Array], factors_true: Sequence[Array]) -> dict[str, Any]:
    errs = [subspace_error(Ah, At) for Ah, At in zip(factors_hat, factors_true)]
    return {"mode_errors": errs, "Err": float(np.mean(errs))}


def latent_posterior_means(
    samples: Sequence[Array],
    mean: Array,
    factors: Sequence[Array],
    *,
    weights: Optional[Array] = None,
    noise_variance: Optional[float] = None,
) -> List[Array]:
    """Compute posterior latent-core means for fitted Tucker loadings."""
    samples, _ = _check_samples(samples)
    ranks = tuple(A.shape[1] for A in factors)
    q = int(np.prod(ranks))
    G = gram_kron(factors)
    out: List[Array] = []
    if weights is None:
        if noise_variance is None:
            weights = np.ones(len(samples), dtype=float)
        else:
            weights = np.full(len(samples), 1.0 / max(float(noise_variance), 1e-12))
    for X, w in zip(samples, weights):
        rhs = multi_mode_dot(X - mean, factors, transpose=True).reshape(-1)
        S = np.linalg.solve(np.eye(q) + float(w) * G, np.eye(q))
        out.append((float(w) * (S @ rhs)).reshape(ranks))
    return out


def reconstruction_errors(samples: Sequence[Array], mean: Array, factors: Sequence[Array], z_means: Sequence[Array]) -> Array:
    errs = []
    for X, Z in zip(samples, z_means):
        R = np.asarray(X, dtype=float) - mean - multi_mode_dot(Z, factors)
        errs.append(float(np.linalg.norm(R) / math.sqrt(R.size)))
    return np.asarray(errs, dtype=float)


@dataclass
class L1TPCAVBEMResult:
    factors: List[Array]
    mean: Array
    rho: float
    weights: Array
    tau_mean: Array
    delta: Array
    z_means: List[Array]
    objective_path: List[float]
    n_iter: int
    converged: bool
    tau0: float = 0.0


class L1TPCAVBEM:
    """Variational EM for robust TPCA with Laplace scale-mixture noise.

    This implements Algorithm 1 from ``L1_PPCA_on_Tensor.pdf``.  The E-step uses
    q(Z_i) Gaussian and q(tau_i) GIG updates.  The M-step is weighted Tucker
    regression; for small tensors the denominator includes the posterior second
    moment C_i,k from Eq. (33), otherwise it falls back to the ECM approximation
    described in Remark 4.3.
    """

    def __init__(
        self,
        ranks: Sequence[int],
        *,
        rho_init: float = 1.0,
        gamma_a: float = 2.0,
        gamma_b: float = 0.5,
        update_rho: bool = True,
        rho_min: Optional[float] = None,
        rho_max: Optional[float] = None,
        rho_damping: float = 1.0,
        monotone_mstep: bool = True,
        max_iter: int = 100,
        tol: float = 1e-4,
        mstep_inner: int = 2,
        ridge: float = 1e-6,
        init: str = "hosvd",
        normalize_factors: bool = True,
        exact_second_moment: bool = True,
        max_second_moment_dim: int = 4000,
        seed: int = 0,
        verbose: bool = False,
        normalize_mode: str = "fixed",
        second_moment_method: str = "kron",
        tau0: float = 0.0,
        tau0_kappa: Optional[float] = None,
    ):
        # Noise floor (revision 2026-11-17): tau_i = tau0 + Exp(rho/2), i.e. the scale
        # prior is truncated below at tau0 > 0.  This bounds the likelihood and the
        # weights (w_i <= 1/tau0).  tau0_kappa, if given, sets
        #     tau0 = tau0_kappa * median_i ||X_i - mean||_F^2 / p   (computed in fit()).
        # tau0 = 0 and tau0_kappa = None reproduce the original model.
        self.tau0 = float(tau0)
        self.tau0_kappa = None if tau0_kappa is None else float(tau0_kappa)
        self.tau0_ = float(tau0)
        # normalize_mode: "fixed" (original; fixes ||A_k||_F = sqrt(n_k m_k), changes L),
        #                 "balance" (revision; L unchanged), "none".
        # second_moment_method: "kron" (revision; exact for any size) or
        #                 "legacy" (original; explicit matrix, falls back to the
        #                 mean-only update when Hk.size*q > max_second_moment_dim).
        self.normalize_mode = str(normalize_mode)
        self.second_moment_method = str(second_moment_method)
        self.ranks = tuple(int(x) for x in ranks)
        self.rho_init = float(rho_init)
        self.gamma_a = float(gamma_a)
        self.gamma_b = float(gamma_b)
        self.update_rho = bool(update_rho)
        self.rho_min = None if rho_min is None else float(rho_min)
        self.rho_max = None if rho_max is None else float(rho_max)
        self.rho_damping = float(rho_damping)
        self.monotone_mstep = bool(monotone_mstep)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.mstep_inner = int(mstep_inner)
        self.ridge = float(ridge)
        self.init = init
        self.normalize_factors = bool(normalize_factors)
        self.exact_second_moment = bool(exact_second_moment)
        self.max_second_moment_dim = int(max_second_moment_dim)
        self.seed = int(seed)
        self.verbose = bool(verbose)

    def _stabilize_rho(self, proposed: float, current: float) -> float:
        proposed = max(float(proposed), 1e-12)
        damping = min(max(self.rho_damping, 0.0), 1.0)
        rho = (1.0 - damping) * max(float(current), 1e-12) + damping * proposed
        if self.rho_min is not None:
            rho = max(rho, self.rho_min)
        if self.rho_max is not None:
            rho = min(rho, self.rho_max)
        return max(rho, 1e-12)

    def _normalize(self, factors: List[Array]) -> List[Array]:
        if not self.normalize_factors:
            return factors
        return normalize_factor_list(factors, self.normalize_mode)

    def _initialize(self, samples: Sequence[Array]) -> Tuple[List[Array], Array]:
        centered, mean = center_samples(samples)
        if self.init == "hosvd":
            init = hosvd_warm_start(centered, self.ranks)
        elif self.init == "random":
            rng = np.random.default_rng(self.seed)
            init = random_factors(samples[0].shape, self.ranks, rng)
        else:
            raise ValueError(f"unknown init={self.init!r}")
        # The initial scale is set as in the original code; "balance" afterwards
        # only redistributes it across modes and leaves the amplitude free.
        return normalize_factor_list(init, "fixed" if self.normalize_factors else "none"), mean

    def _e_step(
        self,
        samples: Sequence[Array],
        mean: Array,
        factors: Sequence[Array],
        weights: Array,
        rho: float,
    ) -> Tuple[List[Array], List[Array], Array, Array, Array, Array]:
        q = int(np.prod(self.ranks))
        p = int(np.prod(samples[0].shape))
        nu = 1.0 - 0.5 * p
        G = gram_kron(factors)
        eye = np.eye(q)
        z_means: List[Array] = []
        z_covs: List[Array] = []
        deltas: List[float] = []

        for X, w in zip(samples, weights):
            centered = X - mean
            rhs = multi_mode_dot(centered, factors, transpose=True).reshape(-1)
            S = np.linalg.solve(eye + float(w) * G, eye)
            z_vec = float(w) * (S @ rhs)
            Z = z_vec.reshape(self.ranks)
            recon = multi_mode_dot(Z, factors)
            resid_sq = float(np.linalg.norm(centered - recon) ** 2)
            delta = resid_sq + float(np.trace(G @ S))
            z_means.append(Z)
            z_covs.append(S)
            deltas.append(max(delta, 1e-12))

        delta_arr = np.asarray(deltas, dtype=float)
        e_tau, e_inv, _ = gig_moments(nu, max(float(rho), 1e-300), delta_arr, lower=self.tau0_)
        tau_mean = np.maximum(e_tau, 1e-300)
        new_weights = np.maximum(e_inv, 1e-300)
        return z_means, z_covs, new_weights, tau_mean, delta_arr, G

    def _update_mean(
        self,
        samples: Sequence[Array],
        z_means: Sequence[Array],
        factors: Sequence[Array],
        weights: Array,
    ) -> Array:
        num = np.zeros_like(samples[0])
        den = max(float(np.sum(weights)), 1e-12)
        for X, Z, w in zip(samples, z_means, weights):
            num += float(w) * (X - multi_mode_dot(Z, factors))
        return num / den

    def _unfold_transform_matrix(self, factors: Sequence[Array], k: int) -> Array:
        q = int(np.prod(self.ranks))
        first = multi_mode_dot(np.zeros(self.ranks), factors, skip=k)
        h_size = unfold(first, k).size
        L = np.empty((h_size, q), dtype=float)
        for j in range(q):
            basis = np.zeros(q, dtype=float)
            basis[j] = 1.0
            H = multi_mode_dot(basis.reshape(self.ranks), factors, skip=k)
            L[:, j] = unfold(H, k).reshape(-1)
        return L

    def _second_moment_C(self, Hk: Array, S: Array, factors: Sequence[Array], k: int) -> Array:
        C = Hk @ Hk.T
        if not self.exact_second_moment:
            return C
        if self.second_moment_method == "kron":
            return C + kron_second_moment_correction(S, factors, self.ranks, k)
        size = Hk.size * int(np.prod(self.ranks))
        if size > self.max_second_moment_dim:
            return C
        L = self._unfold_transform_matrix(factors, k)
        cov_h = L @ S @ L.T
        m_k, cols = Hk.shape
        cov4 = cov_h.reshape(m_k, cols, m_k, cols)
        return C + np.einsum("acbc->ab", cov4)

    def _m_step_factors(
        self,
        samples: Sequence[Array],
        mean: Array,
        z_means: Sequence[Array],
        z_covs: Sequence[Array],
        weights: Array,
        factors: Sequence[Array],
    ) -> List[Array]:
        updated = [np.array(A, dtype=float, copy=True) for A in factors]
        for _ in range(self.mstep_inner):
            for k in range(len(updated)):
                n_k, m_k = updated[k].shape
                lhs = np.zeros((n_k, m_k))
                rhs = self.ridge * np.eye(m_k)
                for X, Z, S, w in zip(samples, z_means, z_covs, weights):
                    H = multi_mode_dot(Z, updated, skip=k)
                    Hk = unfold(H, k)
                    Yk = unfold(X - mean, k)
                    lhs += float(w) * (Yk @ Hk.T)
                    rhs += float(w) * self._second_moment_C(Hk, S, updated, k)
                updated[k] = np.nan_to_num(lhs, nan=0.0, posinf=1e12, neginf=-1e12) @ _stable_pinv(rhs, self.ridge)
            updated = self._normalize(updated)
        return updated

    def _blend_factors(self, old: Sequence[Array], new: Sequence[Array], step: float) -> List[Array]:
        blended = [
            np.asarray(A0, dtype=float) + float(step) * (np.asarray(A1, dtype=float) - np.asarray(A0, dtype=float))
            for A0, A1 in zip(old, new)
        ]
        return self._normalize(blended)

    def _elbo(
        self,
        samples: Sequence[Array],
        z_means: Sequence[Array],
        z_covs: Sequence[Array],
        tau_mean: Array,
        weights: Array,
        deltas: Array,
        rho: float,
    ) -> float:
        """Mean-field ELBO corresponding to Eq. (17), up to no omitted terms.

        The implementation uses q(rho)=Gamma(alpha, beta), whose mean is the
        scalar rho used elsewhere by the E-step:

            alpha = gamma_a + N,
            beta = gamma_b + 0.5 * sum_i E[tau_i],
            E[rho] = alpha / beta.

        For q(tau_i)=GIG(nu, E[rho], delta_i), the ELBO needs E[log tau_i],
        E[tau_i], and E[1/tau_i].  The E-step already provides the latter two;
        E[log tau_i] is evaluated through the GIG Bessel-function identity.
        """
        n = len(samples)
        p = int(np.prod(samples[0].shape))
        q = int(np.prod(self.ranks))
        nu = 1.0 - 0.5 * p
        rho_mean = max(float(rho), 1e-12)
        if self.update_rho:
            alpha = self.gamma_a + n
            beta = self.gamma_b + 0.5 * float(np.sum(np.asarray(tau_mean) - self.tau0_))
            beta = max(beta, 1e-12)
            rho_mean = alpha / beta
            elog_rho = _digamma(alpha) - math.log(beta)
        else:
            alpha = beta = None
            elog_rho = math.log(rho_mean)

        deltas = np.maximum(np.asarray(deltas, dtype=float), 1e-12)
        tau_mean = np.maximum(np.asarray(tau_mean, dtype=float), 1e-12)
        weights = np.maximum(np.asarray(weights, dtype=float), 1e-12)
        elog_tau = gig_expected_log(nu, rho_mean, deltas, lower=self.tau0_)

        log_joint = 0.0
        if self.update_rho:
            log_joint += self.gamma_a * math.log(max(self.gamma_b, 1e-12)) - _gammaln(self.gamma_a)
            log_joint += (self.gamma_a - 1.0) * elog_rho - self.gamma_b * rho_mean
        log_joint += n * (elog_rho - math.log(2.0))

        for Z, S, etau, winv, delta, eltau in zip(z_means, z_covs, tau_mean, weights, deltas, elog_tau):
            ez_sq = float(np.linalg.norm(Z) ** 2 + np.trace(S))
            log_joint += -0.5 * p * math.log(2.0 * math.pi)
            log_joint += -0.5 * p * float(eltau)
            log_joint += -0.5 * float(winv) * float(delta)
            log_joint += -0.5 * q * math.log(2.0 * math.pi)
            log_joint += -0.5 * ez_sq
            log_joint += -0.5 * rho_mean * (float(etau) - self.tau0_)

        entropy_z = 0.0
        for S in z_covs:
            sign, logdet = np.linalg.slogdet(np.asarray(S, dtype=float))
            if sign <= 0:
                logdet = math.log(max(np.linalg.det(S + self.ridge * np.eye(S.shape[0])), 1e-300))
            entropy_z += 0.5 * (q * (1.0 + math.log(2.0 * math.pi)) + float(logdet))

        z_bessel = np.sqrt(rho_mean * deltas)
        # log normaliser of q(tau_i) = -log Z(nu, rho, delta_i); by quadrature (no Bessel overflow)
        log_norm_tau = -gig_log_partition(nu, rho_mean, deltas, lower=self.tau0_)
        elog_q_tau = log_norm_tau + (nu - 1.0) * elog_tau - 0.5 * (rho_mean * tau_mean + deltas * weights)
        entropy_tau = -float(np.sum(elog_q_tau))

        entropy_rho = 0.0
        if self.update_rho:
            assert alpha is not None and beta is not None
            entropy_rho = alpha - math.log(beta) + _gammaln(alpha) + (1.0 - alpha) * _digamma(alpha)

        elbo = log_joint + entropy_z + entropy_tau + entropy_rho
        return float(np.nan_to_num(elbo, nan=-1e300, posinf=1e300, neginf=-1e300))

    def fit(self, samples: Sequence[Array]) -> L1TPCAVBEMResult:
        samples, _ = _check_samples(samples)
        if self.tau0_kappa is not None:
            Xs = np.stack(samples, axis=0)
            dev = (Xs - Xs.mean(axis=0)).reshape(len(samples), -1)
            self.tau0_ = float(self.tau0_kappa * np.median(np.sum(dev * dev, axis=1) / dev.shape[1]))
        else:
            self.tau0_ = self.tau0
        factors, mean = self._initialize(samples)
        rho = max(self.rho_init, 1e-12)
        weights = np.ones(len(samples), dtype=float)
        objective_path: List[float] = []
        prev: Optional[float] = None
        converged = False

        z_means: List[Array] = []
        deltas = np.zeros(len(samples))
        tau_mean = np.ones(len(samples))

        for it in range(1, self.max_iter + 1):
            z_means, z_covs, weights, tau_mean, deltas, G = self._e_step(
                samples, mean, factors, weights, rho
            )
            if self.update_rho:
                shape = self.gamma_a + len(samples)
                rate = self.gamma_b + 0.5 * float(np.sum(np.asarray(tau_mean) - self.tau0_))
                rho = self._stabilize_rho(shape / max(rate, 1e-12), rho)

            z_means, z_covs, weights, tau_mean, deltas, G = self._e_step(
                samples, mean, factors, weights, rho
            )
            current_objective = self._elbo(samples, z_means, z_covs, tau_mean, weights, deltas, rho)

            old_mean = mean
            old_factors = factors
            proposed_mean = self._update_mean(samples, z_means, factors, weights)
            proposed_factors = self._m_step_factors(samples, proposed_mean, z_means, z_covs, weights, factors)

            z_means, z_covs, weights, tau_mean, deltas, G = self._e_step(
                samples, proposed_mean, proposed_factors, weights, rho
            )
            objective = self._elbo(samples, z_means, z_covs, tau_mean, weights, deltas, rho)

            if self.monotone_mstep and objective + 1e-8 < current_objective:
                best = (current_objective, old_mean, old_factors, z_means, z_covs, weights, tau_mean, deltas)
                for step in (0.5, 0.25, 0.125, 0.0625):
                    trial_mean = old_mean + step * (proposed_mean - old_mean)
                    trial_factors = self._blend_factors(old_factors, proposed_factors, step)
                    trial_z, trial_s, trial_w, trial_tau, trial_delta, _ = self._e_step(
                        samples, trial_mean, trial_factors, weights, rho
                    )
                    trial_objective = self._elbo(samples, trial_z, trial_s, trial_tau, trial_w, trial_delta, rho)
                    if trial_objective > best[0]:
                        best = (trial_objective, trial_mean, trial_factors, trial_z, trial_s, trial_w, trial_tau, trial_delta)
                    if trial_objective + 1e-8 >= current_objective:
                        break
                objective, mean, factors, z_means, z_covs, weights, tau_mean, deltas = best
            else:
                mean = proposed_mean
                factors = proposed_factors
            objective_path.append(objective)
            if self.verbose:
                print(
                    f"iter={it:03d} rho={rho:.6g} "
                    f"weight_mean={weights.mean():.6g} objective={objective:.6g}"
                )
            if prev is not None and abs(objective - prev) <= self.tol * (1.0 + abs(prev)):
                converged = True
                break
            prev = objective

        return L1TPCAVBEMResult(
            factors=[A.copy() for A in factors],
            mean=mean.copy(),
            rho=float(rho),
            weights=weights.copy(),
            tau_mean=tau_mean.copy(),
            delta=deltas.copy(),
            tau0=float(self.tau0_),
            z_means=[Z.copy() for Z in z_means],
            objective_path=objective_path,
            n_iter=it,
            converged=converged,
        )


@dataclass
class GaussianTPCAResult:
    factors: List[Array]
    mean: Array
    sigma2: float
    z_means: List[Array]
    objective_path: List[float]
    n_iter: int
    converged: bool


class GaussianTPCAALS:
    """Small Gaussian TPCA EM/ALS baseline using the same Tucker convention."""

    def __init__(
        self,
        ranks: Sequence[int],
        *,
        sigma2_init: Optional[float] = None,
        max_iter: int = 100,
        tol: float = 1e-4,
        mstep_inner: int = 2,
        ridge: float = 1e-6,
        init: str = "hosvd",
        normalize_factors: bool = True,
        seed: int = 0,
        verbose: bool = False,
    ):
        self.ranks = tuple(int(x) for x in ranks)
        self.sigma2_init = sigma2_init
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.mstep_inner = int(mstep_inner)
        self.ridge = float(ridge)
        self.init = init
        self.normalize_factors = bool(normalize_factors)
        self.seed = int(seed)
        self.verbose = bool(verbose)

    def _normalize(self, factors: List[Array]) -> List[Array]:
        if not self.normalize_factors:
            return factors
        out: List[Array] = []
        for A in factors:
            target = math.sqrt(A.shape[0] * A.shape[1])
            out.append(A * (target / max(np.linalg.norm(A, ord="fro"), 1e-12)))
        return out

    def _initialize(self, samples: Sequence[Array]) -> Tuple[List[Array], Array]:
        centered, mean = center_samples(samples)
        if self.init == "hosvd":
            return self._normalize(hosvd_warm_start(centered, self.ranks)), mean
        if self.init == "random":
            rng = np.random.default_rng(self.seed)
            return self._normalize(random_factors(samples[0].shape, self.ranks, rng)), mean
        raise ValueError(f"unknown init={self.init!r}")

    def _e_step(self, samples: Sequence[Array], mean: Array, factors: Sequence[Array], sigma2: float) -> Tuple[List[Array], Array, Array]:
        q = int(np.prod(self.ranks))
        G = gram_kron(factors)
        S = np.linalg.solve(np.eye(q) + G / max(sigma2, 1e-12), np.eye(q))
        z_means: List[Array] = []
        deltas: List[float] = []
        for X in samples:
            rhs = multi_mode_dot(X - mean, factors, transpose=True).reshape(-1)
            z_vec = S @ (rhs / max(sigma2, 1e-12))
            Z = z_vec.reshape(self.ranks)
            R = X - mean - multi_mode_dot(Z, factors)
            z_means.append(Z)
            deltas.append(float(np.linalg.norm(R) ** 2 + np.trace(G @ S)))
        return z_means, np.asarray(deltas, dtype=float), S

    def _update_mean(self, samples: Sequence[Array], z_means: Sequence[Array], factors: Sequence[Array]) -> Array:
        num = np.zeros_like(samples[0])
        for X, Z in zip(samples, z_means):
            num += X - multi_mode_dot(Z, factors)
        return num / max(len(samples), 1)

    def _m_step_factors(self, samples: Sequence[Array], mean: Array, z_means: Sequence[Array], factors: Sequence[Array]) -> List[Array]:
        updated = [np.array(A, dtype=float, copy=True) for A in factors]
        for _ in range(self.mstep_inner):
            for k in range(len(updated)):
                n_k, m_k = updated[k].shape
                lhs = np.zeros((n_k, m_k))
                rhs = self.ridge * np.eye(m_k)
                for X, Z in zip(samples, z_means):
                    Hk = unfold(multi_mode_dot(Z, updated, skip=k), k)
                    Yk = unfold(X - mean, k)
                    lhs += Yk @ Hk.T
                    rhs += Hk @ Hk.T
                updated[k] = lhs @ _stable_pinv(rhs, self.ridge)
            updated = self._normalize(updated)
        return updated

    def fit(self, samples: Sequence[Array]) -> GaussianTPCAResult:
        samples, _ = _check_samples(samples)
        factors, mean = self._initialize(samples)
        if self.sigma2_init is None:
            sigma2 = float(np.var(np.stack(samples, axis=0) - mean))
        else:
            sigma2 = float(self.sigma2_init)
        sigma2 = max(sigma2, 1e-12)
        objective_path: List[float] = []
        prev: Optional[float] = None
        converged = False

        z_means: List[Array] = []
        for it in range(1, self.max_iter + 1):
            z_means, deltas, _ = self._e_step(samples, mean, factors, sigma2)
            mean = self._update_mean(samples, z_means, factors)
            factors = self._m_step_factors(samples, mean, z_means, factors)
            z_means, deltas, _ = self._e_step(samples, mean, factors, sigma2)
            sigma2 = max(float(np.mean(deltas) / np.prod(samples[0].shape)), 1e-12)
            objective = -0.5 * float(np.sum(deltas / sigma2 + np.prod(samples[0].shape) * np.log(sigma2)))
            objective_path.append(objective)
            if self.verbose:
                print(f"iter={it:03d} sigma2={sigma2:.6g} objective={objective:.6g}")
            if prev is not None and abs(objective - prev) <= self.tol * (1.0 + abs(prev)):
                converged = True
                break
            prev = objective

        return GaussianTPCAResult(
            factors=[A.copy() for A in factors],
            mean=mean.copy(),
            sigma2=float(sigma2),
            z_means=[Z.copy() for Z in z_means],
            objective_path=objective_path,
            n_iter=it,
            converged=converged,
        )


# ---------------------------------------------------------------------------
# Corrected Gaussian TPCA baseline (revision 2026-11-17)
# ---------------------------------------------------------------------------
class GaussianTPCAEM:
    """Exact EM for Gaussian TPCA:  vec(X) = vec(mu) + L z + e,  z ~ N(0, I_q),
    e ~ N(0, sigma2 I_p),  L = A_r kron ... kron A_1.

    Differences from the original ``GaussianTPCAALS`` (kept unchanged above for
    reproducibility of Experiments 1-3):
      * the loading update uses the exact posterior second moment
        C = Hbar Hbar^T + V (``kron_second_moment_correction``), so every block
        update maximises the expected complete-data log-likelihood (ECM);
      * the E-step solves with (sigma2 I + G) rather than (I + G/sigma2), which
        stays well conditioned when sigma2 is small;
      * the exact marginal log-likelihood is computed (Woodbury) and checked to
        be nondecreasing; a decrease larger than ``monotone_tol`` stops the fit
        and is reported (``monotone_violation``);
      * sigma2 has a relative floor (``sigma2_floor`` times the data variance);
      * factor normalisation defaults to "balance", which leaves L unchanged.
    """

    def __init__(
        self,
        ranks: Sequence[int],
        *,
        max_iter: int = 100,
        tol: float = 1e-6,
        mstep_inner: int = 1,
        ridge: float = 1e-10,
        normalize_mode: str = "balance",
        sigma2_floor: float = 1e-8,
        monotone_tol: float = 1e-7,
        seed: int = 0,
        verbose: bool = False,
    ):
        self.ranks = tuple(int(x) for x in ranks)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.mstep_inner = int(mstep_inner)
        self.ridge = float(ridge)
        self.normalize_mode = str(normalize_mode)
        self.sigma2_floor = float(sigma2_floor)
        self.monotone_tol = float(monotone_tol)
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.monotone_violation = False

    def _e_step(self, samples, mean, factors, sigma2):
        q = int(np.prod(self.ranks))
        G = gram_kron(factors)
        M = sigma2 * np.eye(q) + G
        Minv = np.linalg.solve(M, np.eye(q))
        S = sigma2 * Minv  # posterior covariance, shared by all samples
        z_means, rtil = [], []
        for X in samples:
            r = multi_mode_dot(X - mean, factors, transpose=True).reshape(-1)
            z_means.append((Minv @ r).reshape(self.ranks))
            rtil.append(r)
        return z_means, S, G, Minv, rtil

    def loglik(self, samples, mean, factors, sigma2) -> float:
        p = int(np.prod(samples[0].shape))
        q = int(np.prod(self.ranks))
        G = gram_kron(factors)
        M = sigma2 * np.eye(q) + G
        Minv = np.linalg.solve(M, np.eye(q))
        _, logdetM = np.linalg.slogdet(M)
        # log|L L^T + s I_p| = (p - q) log s + log|s I_q + G|
        logdet = (p - q) * math.log(sigma2) + logdetM
        total = 0.0
        for X in samples:
            R = X - mean
            r = multi_mode_dot(R, factors, transpose=True).reshape(-1)
            quad = (float(np.sum(R * R)) - float(r @ Minv @ r)) / sigma2
            total += -0.5 * (p * math.log(2.0 * math.pi) + logdet + quad)
        return float(total)

    def fit(self, samples: Sequence[Array]) -> GaussianTPCAResult:
        samples, shape = _check_samples(samples)
        p = int(np.prod(shape))
        centered, mean = center_samples(samples)
        factors = normalize_factor_list(hosvd_warm_start(centered, self.ranks), "fixed")
        data_var = float(np.var(np.stack(samples, axis=0) - mean))
        floor = max(self.sigma2_floor * data_var, 1e-300)
        sigma2 = max(data_var, floor)
        path: List[float] = [self.loglik(samples, mean, factors, sigma2)]
        converged = False
        it = 0
        z_means: List[Array] = []
        for it in range(1, self.max_iter + 1):
            z_means, S, G, _, _ = self._e_step(samples, mean, factors, sigma2)
            # M-step (ECM blocks): mean, then each loading, then sigma2
            mean = np.mean(np.stack([X - multi_mode_dot(Z, factors) for X, Z in zip(samples, z_means)]), axis=0)
            for _ in range(self.mstep_inner):
                for k in range(len(factors)):
                    n_k, m_k = factors[k].shape
                    lhs = np.zeros((n_k, m_k))
                    rhs = self.ridge * np.eye(m_k)
                    Vk = kron_second_moment_correction(S, factors, self.ranks, k)
                    for X, Z in zip(samples, z_means):
                        Hk = unfold(multi_mode_dot(Z, factors, skip=k), k)
                        lhs += unfold(X - mean, k) @ Hk.T
                        rhs += Hk @ Hk.T + Vk
                    factors[k] = np.linalg.solve(rhs.T, lhs.T).T
                factors = normalize_factor_list(factors, self.normalize_mode)
            G = gram_kron(factors)
            resid = 0.0
            for X, Z in zip(samples, z_means):
                R = X - mean - multi_mode_dot(Z, factors)
                resid += float(np.sum(R * R))
            sigma2 = max((resid + len(samples) * float(np.trace(G @ S))) / (len(samples) * p), floor)
            ll = self.loglik(samples, mean, factors, sigma2)
            if self.verbose:
                print(f"  [GaussEM] iter={it:03d} sigma2={sigma2:.6g} loglik={ll:.8g}", flush=True)
            if ll < path[-1] - self.monotone_tol * (1.0 + abs(path[-1])):
                self.monotone_violation = True
                path.append(ll)
                break
            converged = abs(ll - path[-1]) <= self.tol * (1.0 + abs(path[-1]))
            path.append(ll)
            if converged:
                break
        z_means, S, _, _, _ = self._e_step(samples, mean, factors, sigma2)
        return GaussianTPCAResult(
            factors=[A.copy() for A in factors],
            mean=mean.copy(),
            sigma2=float(sigma2),
            z_means=[Z.copy() for Z in z_means],
            objective_path=path,
            n_iter=it,
            converged=converged,
        )
