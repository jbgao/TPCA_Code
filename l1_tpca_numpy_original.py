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


def gram_kron(factors: Sequence[Array]) -> Array:
    """Kronecker product A_r^T A_r x ... x A_1^T A_1."""
    out = np.array([[1.0]])
    for A in reversed(factors):
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


def gig_moment(nu: float, a: float, b: Array | float, order: float) -> Array:
    """E[X^order] for GIG(nu, a, b): x^(nu-1) exp(-(a*x + b/x)/2)."""
    b_arr = np.maximum(np.asarray(b, dtype=float), 1e-12)
    a = max(float(a), 1e-12)
    z = np.sqrt(a * b_arr)
    moment = (b_arr / a) ** (0.5 * order) * _k_ratio(nu + order, nu, z)
    return np.nan_to_num(moment, nan=1e-12, posinf=1e12, neginf=1e-12)


def gig_expected_log(nu: float, a: float, b: Array | float) -> Array:
    """E[log X] for GIG(nu, a, b).

    The identity is
        E log X = 0.5 log(b/a) + d/dnu log K_nu(sqrt(a b)).
    We evaluate the order derivative by a centred finite difference. This keeps
    the objective aligned with the VB updates without adding a hard scipy
    dependency beyond the optional Bessel helper already used above.
    """
    b_arr = np.maximum(np.asarray(b, dtype=float), 1e-12)
    a = max(float(a), 1e-12)
    z = np.sqrt(a * b_arr)
    h = 1e-4 * max(1.0, abs(float(nu)))
    dlog_k = (_log_kv(nu + h, z) - _log_kv(nu - h, z)) / (2.0 * h)
    out = 0.5 * np.log(b_arr / a) + dlog_k
    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=-1e12)


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
    ):
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
        out: List[Array] = []
        for A in factors:
            target = math.sqrt(A.shape[0] * A.shape[1])
            fro = np.linalg.norm(A, ord="fro")
            out.append(A * (target / max(fro, 1e-12)))
        return out

    def _initialize(self, samples: Sequence[Array]) -> Tuple[List[Array], Array]:
        centered, mean = center_samples(samples)
        if self.init == "hosvd":
            return self._normalize(hosvd_warm_start(centered, self.ranks)), mean
        if self.init == "random":
            rng = np.random.default_rng(self.seed)
            return self._normalize(random_factors(samples[0].shape, self.ranks, rng)), mean
        raise ValueError(f"unknown init={self.init!r}")

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
        tau_mean = np.clip(gig_moment(nu, rho, delta_arr, 1.0), 1e-12, 1e12)
        new_weights = np.clip(gig_moment(nu, rho, delta_arr, -1.0), 1e-12, 1e12)
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
            beta = self.gamma_b + 0.5 * float(np.sum(tau_mean))
            beta = max(beta, 1e-12)
            rho_mean = alpha / beta
            elog_rho = _digamma(alpha) - math.log(beta)
        else:
            alpha = beta = None
            elog_rho = math.log(rho_mean)

        deltas = np.maximum(np.asarray(deltas, dtype=float), 1e-12)
        tau_mean = np.maximum(np.asarray(tau_mean, dtype=float), 1e-12)
        weights = np.maximum(np.asarray(weights, dtype=float), 1e-12)
        elog_tau = gig_expected_log(nu, rho_mean, deltas)

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
            log_joint += -0.5 * rho_mean * float(etau)

        entropy_z = 0.0
        for S in z_covs:
            sign, logdet = np.linalg.slogdet(np.asarray(S, dtype=float))
            if sign <= 0:
                logdet = math.log(max(np.linalg.det(S + self.ridge * np.eye(S.shape[0])), 1e-300))
            entropy_z += 0.5 * (q * (1.0 + math.log(2.0 * math.pi)) + float(logdet))

        z_bessel = np.sqrt(rho_mean * deltas)
        log_norm_tau = 0.5 * nu * (math.log(rho_mean) - np.log(deltas)) - math.log(2.0) - _log_kv(nu, z_bessel)
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
                rate = self.gamma_b + 0.5 * float(np.sum(tau_mean))
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
