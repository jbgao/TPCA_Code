from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Any
import math
import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import orthogonal_procrustes


Array = np.ndarray


# -----------------------------------------------------------------------------
# Tensor utilities
# -----------------------------------------------------------------------------


def unfold(X: Array, mode: int) -> Array:
    """Mode-k unfolding with shape (n_k, prod_{j!=k} n_j)."""
    X = np.asarray(X)
    order = (mode,) + tuple(i for i in range(X.ndim) if i != mode)
    return np.transpose(X, order).reshape(X.shape[mode], -1)



def fold(Xk: Array, shape: Sequence[int], mode: int) -> Array:
    """Inverse of unfold."""
    shape = tuple(shape)
    order = (mode,) + tuple(i for i in range(len(shape)) if i != mode)
    inv_order = np.argsort(order)
    full = Xk.reshape([shape[mode]] + [shape[i] for i in range(len(shape)) if i != mode])
    return np.transpose(full, inv_order)



def mode_dot(X: Array, A: Array, mode: int) -> Array:
    """Compute X ×_mode A, where A has shape (p, n_mode)."""
    X = np.asarray(X)
    A = np.asarray(A)
    Xk = unfold(X, mode)
    Yk = A @ Xk
    new_shape = list(X.shape)
    new_shape[mode] = A.shape[0]
    return fold(Yk, new_shape, mode)



def multi_mode_dot(X: Array, factors: Sequence[Array], skip: Optional[int] = None,
                   transpose: bool = False) -> Array:
    """Apply Tucker multilinear map.

    If transpose is False, apply A_k on mode k. Otherwise apply A_k^T.
    If skip is not None, skip that mode.
    """
    Y = np.asarray(X)
    for k, A in enumerate(factors):
        if skip is not None and k == skip:
            continue
        M = A.T if transpose else A
        Y = mode_dot(Y, M, k)
    return Y



def kron_factors(factors: Sequence[Array]) -> Array:
    """Kronecker product A_r ⊗ ... ⊗ A_1 in the paper's vectorized convention."""
    out = np.array([[1.0]])
    for A in reversed(factors):
        out = np.kron(out, A)
    return out



def gram_kron(factors: Sequence[Array]) -> Array:
    """Kronecker product of Gram matrices A_r^T A_r ⊗ ... ⊗ A_1^T A_1."""
    out = np.array([[1.0]])
    for A in reversed(factors):
        out = np.kron(out, A.T @ A)
    return out



def center_samples(samples: Sequence[Array]) -> Tuple[List[Array], Array]:
    mean = np.mean(np.stack(samples, axis=0), axis=0)
    centered = [np.asarray(X) - mean for X in samples]
    return centered, mean



def _check_samples(samples: Sequence[Array]) -> Tuple[List[Array], Tuple[int, ...]]:
    if len(samples) == 0:
        raise ValueError("samples must be non-empty")
    arrs = [np.asarray(X, dtype=float) for X in samples]
    shape = arrs[0].shape
    if any(X.shape != shape for X in arrs):
        raise ValueError("all samples must have the same shape")
    return arrs, shape


# -----------------------------------------------------------------------------
# Data generation and metrics
# -----------------------------------------------------------------------------


def random_factors(shape: Sequence[int], ranks: Sequence[int], rng: np.random.Generator,
                   scale_like_paper: bool = True) -> List[Array]:
    """Generate factors as in Case 1 of the paper.

    Draw A_raw ~ N(0,1), take its SVD UDV^T and set A = U D, then rescale so that
    ||A||_F = sqrt(n_k m_k).
    """
    factors = []
    for n, m in zip(shape, ranks):
        Araw = rng.normal(size=(n, m))
        U, s, _ = np.linalg.svd(Araw, full_matrices=False)
        A = U[:, :m] @ np.diag(s[:m])
        if scale_like_paper:
            target = math.sqrt(n * m)
            A = A * (target / np.linalg.norm(A, ord="fro"))
        factors.append(A)
    return factors



def sample_tpca(factors: Sequence[Array], sigma2: float, n_samples: int,
                rng: np.random.Generator, mean: Optional[Array] = None) -> List[Array]:
    shape = tuple(A.shape[0] for A in factors)
    ranks = tuple(A.shape[1] for A in factors)
    mean_tensor = np.zeros(shape) if mean is None else np.asarray(mean, dtype=float)
    samples = []
    for _ in range(n_samples):
        Z = rng.normal(size=ranks)
        X = multi_mode_dot(Z, factors) + rng.normal(scale=math.sqrt(sigma2), size=shape) + mean_tensor
        samples.append(X)
    return samples



def factor_error(A_hat: Array, A_true: Array) -> float:
    """Procrustes-aligned RMSE normalized by sqrt(n_k m_k)."""
    Q, _ = orthogonal_procrustes(A_hat, A_true)
    return np.linalg.norm(A_hat @ Q - A_true, ord="fro") / math.sqrt(np.prod(A_true.shape))



def overall_factor_error(factors_hat: Sequence[Array], factors_true: Sequence[Array]) -> Dict[str, Any]:
    errs = [factor_error(Ah, At) for Ah, At in zip(factors_hat, factors_true)]
    return {"mode_errors": errs, "Err": float(np.mean(errs))}


# -----------------------------------------------------------------------------
# HOSVD warm start used by the paper's EM experiments
# -----------------------------------------------------------------------------


def hosvd_warm_start(samples: Sequence[Array], ranks: Sequence[int],
                     scale_to_sqrt_nm: bool = True) -> List[Array]:
    samples, shape = _check_samples(samples)
    factors = []
    for k, (n, m) in enumerate(zip(shape, ranks)):
        # Stack mode-k unfoldings horizontally across all samples.
        stacked = np.concatenate([unfold(X, k) for X in samples], axis=1)
        U, _, _ = np.linalg.svd(stacked, full_matrices=False)
        A = U[:, :m]
        if scale_to_sqrt_nm:
            A = A * math.sqrt(n)
        factors.append(A)
    return factors


# -----------------------------------------------------------------------------
# EM algorithm (paper-faithful, practical reimplementation)
# -----------------------------------------------------------------------------


@dataclass
class TPCAEMResult:
    factors: List[Array]
    sigma2: float
    mean: Array
    nu: Optional[Array]
    loglik_path: List[float]
    n_iter: int
    converged: bool


class TPCAEM:
    """Practical reimplementation of the paper's EM algorithm.

    Notes
    -----
    - The paper's experiments in Section 6 use mean tensor nu = 0. This implementation
      defaults to a centered / zero-mean fit and is faithful to that setup.
    - A structured latent mean nu can optionally be updated, but it is not used by default.
    - The E-step follows eqs. (15)–(18), and the M-step uses eq. (19), eq. (21), and the
      closed-form mode update from Appendix B eq. (40). See the paper for the operator-form
      derivation.
    """

    def __init__(self,
                 ranks: Sequence[int],
                 tol: float = 1e-3,
                 max_iter: int = 100,
                 sigma2_init: Optional[float] = None,
                 init: str = "hosvd",
                 update_nu: bool = False,
                 center: bool = True,
                 verbose: bool = False):
        self.ranks = tuple(int(x) for x in ranks)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.sigma2_init = sigma2_init
        self.init = init
        self.update_nu = bool(update_nu)
        self.center = bool(center)
        self.verbose = bool(verbose)

    @staticmethod
    def _compute_Wk(V: Array, factors: Sequence[Array], k: int) -> Array:
        """Compute W_k from pair(V) by reshaping + contractions.

        V is q x q with q = prod m_k. Reshape to pair(V) of shape (m_1^2, ..., m_r^2),
        then contract with vec(A_j^T A_j) for j != k.
        """
        ranks = [A.shape[1] for A in factors]
        r = len(ranks)
        V_tensor = V.reshape(tuple(ranks) + tuple(ranks))
        # interleave latent row/col mode indices: (a1,b1,a2,b2,...)
        perm = []
        for j in range(r):
            perm.extend([j, j + r])
        pairV = np.transpose(V_tensor, perm).reshape([m * m for m in ranks])

        out = pairV
        current_axes = list(range(r))
        # Contract in descending axis order to keep indexing stable.
        for j in sorted([j for j in range(r) if j != k], reverse=True):
            Gj = factors[j].T @ factors[j]
            out = np.tensordot(out, Gj.reshape(-1), axes=([j], [0]))
            current_axes.pop(j)
        if out.ndim != 1 or out.size != ranks[k] * ranks[k]:
            raise RuntimeError("unexpected shape when building W_k")
        return out.reshape(ranks[k], ranks[k])

    def _loglik(self, samples: Sequence[Array], mean: Array, factors: Sequence[Array], sigma2: float) -> float:
        M = kron_factors(factors)
        p, q = M.shape
        G = M.T @ M
        K = G + sigma2 * np.eye(q)
        sign, logdetK = np.linalg.slogdet(K)
        if sign <= 0:
            return -np.inf
        logdetSigma = p * np.log(sigma2) + np.linalg.slogdet(np.eye(q) + G / sigma2)[1]
        # Equivalent, but the line above is numerically okay. Use K-based identity to stay stable.
        logdetSigma = (p - q) * np.log(sigma2) + logdetK
        Kinv = np.linalg.inv(K)
        ll = 0.0
        for X in samples:
            x = (X - mean).reshape(-1)
            quad = (x @ x) / sigma2 - (x @ M @ Kinv @ M.T @ x) / sigma2
            quad = quad / sigma2
            ll += -0.5 * (p * np.log(2.0 * np.pi) + logdetSigma + quad)
        return float(ll)

    def fit(self, samples: Sequence[Array]) -> TPCAEMResult:
        samples, shape = _check_samples(samples)
        N = len(samples)
        p = int(np.prod(shape))
        ranks = self.ranks
        q = int(np.prod(ranks))

        if self.center:
            centered, mean = center_samples(samples)
        else:
            centered = list(samples)
            mean = np.zeros(shape)

        if self.init == "hosvd":
            factors = hosvd_warm_start(centered, ranks)
        else:
            raise ValueError(f"unknown init={self.init!r}")

        if self.sigma2_init is None:
            total_var = np.mean([np.mean(X ** 2) for X in centered])
            sigma2 = max(1e-6, 0.1 * total_var)
        else:
            sigma2 = float(self.sigma2_init)

        nu = np.zeros(ranks)
        loglik_path: List[float] = []
        converged = False

        for it in range(1, self.max_iter + 1):
            M_load = kron_factors(factors)
            G = gram_kron(factors)
            M_small = G + sigma2 * np.eye(q)
            M_small_inv = np.linalg.inv(M_small)

            # E-step: E_i and common V.
            V = sigma2 * M_small_inv
            Eis: List[Array] = []
            for X in centered:
                x = (X - multi_mode_dot(nu, factors)).reshape(-1)
                Ez = M_small_inv @ (M_load.T @ x)
                Eis.append(Ez.reshape(ranks))

            # M-step for A_k via Appendix B, eq. (40).
            updated = list(factors)
            for k in range(len(shape)):
                Wk = self._compute_Wk(V, factors, k)
                XtU = np.zeros((shape[k], ranks[k]))
                UUt = np.zeros((ranks[k], ranks[k]))
                for X, Ei in zip(centered, Eis):
                    latent_shift = nu + Ei
                    Ui = unfold(multi_mode_dot(latent_shift, factors, skip=k), k)
                    XtU += unfold(X, k) @ Ui.T
                    UUt += Ui @ Ui.T
                XtU /= N
                UUt /= N
                updated[k] = XtU @ np.linalg.inv(Wk + UUt)

            factors = updated
            M_load = kron_factors(factors)
            G = gram_kron(factors)

            if self.update_nu:
                avg_term = np.mean(
                    [X.reshape(-1) - M_load @ Ei.reshape(-1) for X, Ei in zip(centered, Eis)], axis=0
                )
                nu_vec = np.linalg.solve(G, M_load.T @ avg_term)
                nu = nu_vec.reshape(ranks)
            else:
                nu = np.zeros(ranks)

            mu_tensor = multi_mode_dot(nu, factors)
            residual_ss = 0.0
            for X, Ei in zip(centered, Eis):
                recon = mu_tensor + multi_mode_dot(Ei, factors)
                residual_ss += np.linalg.norm(X - recon) ** 2
            tr_term = float(np.trace(M_load @ V @ M_load.T))
            sigma2_new = max(1e-10, residual_ss / (p * N) + tr_term / p)
            sigma2 = sigma2_new

            ll = self._loglik(centered, np.zeros(shape), factors, sigma2) if not self.update_nu else self._loglik(centered, mu_tensor, factors, sigma2)
            loglik_path.append(ll)
            if self.verbose:
                print(f"[EM] iter={it:03d} sigma2={sigma2:.6g} loglik={ll:.6g}")

            if it > 1 and abs(loglik_path[-1] - loglik_path[-2]) <= self.tol * (1.0 + abs(loglik_path[-2])):
                converged = True
                break

        total_mean = mean + multi_mode_dot(nu, factors)
        return TPCAEMResult(
            factors=factors,
            sigma2=sigma2,
            mean=total_mean,
            nu=nu if self.update_nu else None,
            loglik_path=loglik_path,
            n_iter=it,
            converged=converged,
        )


# -----------------------------------------------------------------------------
# Rank-1 power iteration (practical and scalable reimplementation)
# -----------------------------------------------------------------------------


@dataclass
class TPCAPowerResult:
    factors: List[Array]
    B_mats: List[Array]
    sigma2: float
    omega: float
    n_iter: int


class TPCARank1Power:
    """Practical reimplementation of Algorithm 1.

    This version does not materialize pair(S_N) explicitly. Instead it uses the sample-wise
    projection identity from Sec. 5: the mode-k update can be computed from projected sample
    covariances, which keeps memory manageable for the large-dimension experiments.
    """

    def __init__(self,
                 ranks: Sequence[int],
                 n_iter: int = 5,
                 init: str = "random_psd",
                 seed: Optional[int] = None):
        self.ranks = tuple(int(x) for x in ranks)
        self.n_iter = int(n_iter)
        self.init = init
        self.seed = seed

    def _init_B(self, shape: Sequence[int], rng: np.random.Generator) -> List[Array]:
        B_list = []
        for n, m in zip(shape, self.ranks):
            if self.init == "random_psd":
                H = rng.normal(size=(n, m))
                B = H @ H.T
                B /= np.linalg.norm(B, ord="fro")
            elif self.init == "hosvd":
                raise ValueError("hosvd init is not supported in _init_B; pass factors explicitly if needed")
            else:
                raise ValueError(f"unknown init={self.init!r}")
            B_list.append(B)
        return B_list

    @staticmethod
    def _sqrt_factor(B: Array, rank_cap: int) -> Array:
        evals, evecs = np.linalg.eigh((B + B.T) / 2.0)
        idx = np.argsort(evals)[::-1]
        evals = np.clip(evals[idx[:rank_cap]], 0.0, None)
        evecs = evecs[:, idx[:rank_cap]]
        return evecs * np.sqrt(evals)[None, :]

    @staticmethod
    def _top_rank_psd(B: Array, rank_cap: int) -> Array:
        evals, evecs = np.linalg.eigh((B + B.T) / 2.0)
        idx = np.argsort(evals)[::-1][:rank_cap]
        evals = np.clip(evals[idx], 0.0, None)
        B_rank = (evecs[:, idx] * evals[None, :]) @ evecs[:, idx].T
        fro = np.linalg.norm(B_rank, ord="fro")
        if fro == 0:
            return np.eye(B.shape[0]) / np.sqrt(B.shape[0])
        return B_rank / fro

    @staticmethod
    def _tilde_Bk(samples: Sequence[Array], B_list: Sequence[Array], k: int, ranks: Sequence[int]) -> Array:
        """Efficient form of eq. (25) using projected sample covariances.

        If B_j = H_j H_j^T for j != k, then
            \tilde B_k = (1/N) sum_i Y_{i,(k)} Y_{i,(k)}^T,
        where Y_i = X_i ×_{j != k} H_j^T.
        """
        H_list = [TPCARank1Power._sqrt_factor(Bj, rank_cap=rj) for Bj, rj in zip(B_list, ranks)]
        n_k = samples[0].shape[k]
        out = np.zeros((n_k, n_k))
        for X in samples:
            Y = np.asarray(X)
            for j, H in enumerate(H_list):
                if j == k:
                    continue
                Y = mode_dot(Y, H.T, j)
            Yk = unfold(Y, k)
            out += Yk @ Yk.T
        out /= len(samples)
        return out

    @staticmethod
    def _omega_hat(samples: Sequence[Array], B_list: Sequence[Array], ranks: Sequence[int]) -> float:
        H_list = [TPCARank1Power._sqrt_factor(Bj, rank_cap=rj) for Bj, rj in zip(B_list, ranks)]
        vals = []
        for X in samples:
            Y = np.asarray(X)
            for j, H in enumerate(H_list):
                Y = mode_dot(Y, H.T, j)
            vals.append(float(np.linalg.norm(Y) ** 2))
        return float(np.mean(vals))

    def fit(self, samples: Sequence[Array], B_init: Optional[Sequence[Array]] = None) -> TPCAPowerResult:
        samples, shape = _check_samples(samples)
        rng = np.random.default_rng(self.seed)
        B_list = [np.asarray(B, dtype=float) for B in B_init] if B_init is not None else self._init_B(shape, rng)

        for _ in range(self.n_iter):
            new_B = []
            for k in range(len(shape)):
                B_tilde = self._tilde_Bk(samples, B_list, k, self.ranks)
                new_B.append(self._top_rank_psd(B_tilde, self.ranks[k]))
            B_list = new_B

        omega_hat = self._omega_hat(samples, B_list, self.ranks)
        scale = omega_hat ** (1.0 / len(shape))

        A_hat = []
        B_hat_scaled = []
        for k, B in enumerate(B_list):
            B_scaled = scale * B
            evals, evecs = np.linalg.eigh((B_scaled + B_scaled.T) / 2.0)
            idx = np.argsort(evals)[::-1][: self.ranks[k]]
            evals = np.clip(evals[idx], 0.0, None)
            evecs = evecs[:, idx]
            A_hat.append(evecs * np.sqrt(evals)[None, :])
            B_hat_scaled.append(B_scaled)

        mean_sq_norm = float(np.mean([np.linalg.norm(X) ** 2 for X in samples]))
        tr_prod = 1.0
        for B in B_hat_scaled:
            tr_prod *= float(np.trace(B))
        n = int(np.prod(shape))
        sigma2_hat = max(0.0, mean_sq_norm / n - tr_prod / n)

        return TPCAPowerResult(
            factors=A_hat,
            B_mats=B_hat_scaled,
            sigma2=sigma2_hat,
            omega=omega_hat,
            n_iter=self.n_iter,
        )


# -----------------------------------------------------------------------------
# Experiments from Section 6
# -----------------------------------------------------------------------------


def case1_em_once(n: int, m: int, sigma2: float, rng: np.random.Generator,
                  n_samples: int = 15, tol: float = 1e-3, max_iter: int = 100) -> Dict[str, Any]:
    shape = (n, n, n)
    ranks = (m, m, m)
    true_factors = random_factors(shape, ranks, rng)
    samples = sample_tpca(true_factors, sigma2=sigma2, n_samples=n_samples, rng=rng)
    model = TPCAEM(ranks=ranks, tol=tol, max_iter=max_iter, sigma2_init=sigma2,
                   init="hosvd", update_nu=False, center=True, verbose=False)
    fit = model.fit(samples)
    err = overall_factor_error(fit.factors, true_factors)
    return {
        "shape": shape,
        "ranks": ranks,
        "sigma2_true": sigma2,
        "sigma2_hat": fit.sigma2,
        "Err": err["Err"],
        "mode_errors": err["mode_errors"],
        "n_iter": fit.n_iter,
        "converged": fit.converged,
        "loglik_path": fit.loglik_path,
    }



def case2_power_once(rng: np.random.Generator, n_iter: int) -> Dict[str, Any]:
    shape = (15, 15, 15)
    ranks = (3, 3, 3)
    sigma2 = 1.0
    N = 400
    true_factors = random_factors(shape, ranks, rng)
    samples = sample_tpca(true_factors, sigma2=sigma2, n_samples=N, rng=rng)
    model = TPCARank1Power(ranks=ranks, n_iter=n_iter, seed=int(rng.integers(0, 2**31 - 1)))
    fit = model.fit(samples)
    err = overall_factor_error(fit.factors, true_factors)
    return {
        "shape": shape,
        "ranks": ranks,
        "sigma2_true": sigma2,
        "sigma2_hat": fit.sigma2,
        "Err": err["Err"],
        "mode_errors": err["mode_errors"],
        "n_iter": n_iter,
    }



def case3_power_once(shape: Sequence[int], ranks: Sequence[int], sigma2: float, N: int,
                     rng: np.random.Generator, n_iter: int = 5) -> Dict[str, Any]:
    true_factors = random_factors(shape, ranks, rng)
    samples = sample_tpca(true_factors, sigma2=sigma2, n_samples=N, rng=rng)
    model = TPCARank1Power(ranks=ranks, n_iter=n_iter, seed=int(rng.integers(0, 2**31 - 1)))
    fit = model.fit(samples)
    err = overall_factor_error(fit.factors, true_factors)
    rel_sigma_err = abs(fit.sigma2 - sigma2) / sigma2
    return {
        "shape": tuple(shape),
        "ranks": tuple(ranks),
        "sigma2_true": sigma2,
        "sigma2_hat": fit.sigma2,
        "sigma2_relerr": rel_sigma_err,
        "Err": err["Err"],
        "mode_errors": err["mode_errors"],
    }


__all__ = [
    "TPCAEM",
    "TPCAEMResult",
    "TPCARank1Power",
    "TPCAPowerResult",
    "random_factors",
    "sample_tpca",
    "hosvd_warm_start",
    "factor_error",
    "overall_factor_error",
    "case1_em_once",
    "case2_power_once",
    "case3_power_once",
]
