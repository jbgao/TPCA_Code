from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, Optional, List, Dict, Any
import math
import numpy as np
import torch
from scipy.special import kve

Tensor = torch.Tensor


# -----------------------------------------------------------------------------
# Device / dtype helpers
# -----------------------------------------------------------------------------


def resolve_device(device: Optional[str | torch.device] = None) -> torch.device:
    if device is None or str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)



def _to_tensor(x: Any, *, device: torch.device, dtype: torch.dtype = torch.float64) -> Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)



def _to_numpy(x: Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


# -----------------------------------------------------------------------------
# Tensor utilities
# -----------------------------------------------------------------------------


def unfold(X: Tensor | np.ndarray, mode: int) -> Tensor:
    X = torch.as_tensor(X)
    order = (mode,) + tuple(i for i in range(X.ndim) if i != mode)
    return X.permute(order).reshape(X.shape[mode], -1)



def fold(Xk: Tensor | np.ndarray, shape: Sequence[int], mode: int) -> Tensor:
    Xk = torch.as_tensor(Xk)
    shape = tuple(shape)
    order = (mode,) + tuple(i for i in range(len(shape)) if i != mode)
    inv_order = tuple(np.argsort(order).tolist())
    full = Xk.reshape([shape[mode]] + [shape[i] for i in range(len(shape)) if i != mode])
    return full.permute(inv_order)



def mode_dot(X: Tensor | np.ndarray, A: Tensor | np.ndarray, mode: int) -> Tensor:
    X = torch.as_tensor(X)
    A = torch.as_tensor(A)
    Xk = unfold(X, mode)
    Yk = A @ Xk
    new_shape = list(X.shape)
    new_shape[mode] = A.shape[0]
    return fold(Yk, new_shape, mode)



def multi_mode_dot(
    X: Tensor | np.ndarray,
    factors: Sequence[Tensor | np.ndarray],
    skip: Optional[int] = None,
    transpose: bool = False,
) -> Tensor:
    Y = torch.as_tensor(X)
    for k, A in enumerate(factors):
        if skip is not None and k == skip:
            continue
        A = torch.as_tensor(A)
        M = A.transpose(0, 1) if transpose else A
        Y = mode_dot(Y, M, k)
    return Y



def gram_kron(factors: Sequence[Tensor | np.ndarray]) -> Tensor:
    out = torch.ones((1, 1), dtype=torch.float64, device=torch.as_tensor(factors[0]).device)
    for A in reversed(factors):
        A = torch.as_tensor(A)
        out = torch.kron(out, A.transpose(0, 1) @ A)
    return out



def center_samples(samples: Sequence[Tensor | np.ndarray]) -> Tuple[List[Tensor], Tensor]:
    stack = torch.stack([torch.as_tensor(X) for X in samples], dim=0)
    mean = stack.mean(dim=0)
    centered = [torch.as_tensor(X) - mean for X in samples]
    return centered, mean



def _check_samples(
    samples: Sequence[Tensor | np.ndarray], *, device: torch.device, dtype: torch.dtype = torch.float64
) -> Tuple[List[Tensor], Tuple[int, ...]]:
    if len(samples) == 0:
        raise ValueError("samples must be non-empty")
    arrs = [_to_tensor(X, device=device, dtype=dtype) for X in samples]
    shape = tuple(arrs[0].shape)
    if any(tuple(X.shape) != shape for X in arrs):
        raise ValueError("all samples must have the same shape")
    return arrs, shape



def _randn(shape: Sequence[int], rng: np.random.Generator, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.from_numpy(rng.normal(size=tuple(shape))).to(device=device, dtype=dtype)



def random_factors(
    shape: Sequence[int],
    ranks: Sequence[int],
    rng: np.random.Generator,
    scale_like_paper: bool = True,
    device: Optional[str | torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> List[np.ndarray]:
    device = resolve_device(device)
    factors: List[np.ndarray] = []
    for n, m in zip(shape, ranks):
        Araw = _randn((n, m), rng, device=device, dtype=dtype)
        U, s, _ = torch.linalg.svd(Araw, full_matrices=False)
        A = U[:, :m] @ torch.diag(s[:m])
        if scale_like_paper:
            target = math.sqrt(n * m)
            A = A * (target / torch.linalg.norm(A, ord="fro"))
        factors.append(_to_numpy(A))
    return factors



def hosvd_warm_start(
    samples: Sequence[Tensor | np.ndarray],
    ranks: Sequence[int],
    scale_to_sqrt_nm: bool = True,
    device: Optional[str | torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> List[np.ndarray]:
    device = resolve_device(device)
    samples_t, shape = _check_samples(samples, device=device, dtype=dtype)
    factors: List[np.ndarray] = []
    for k, (n, m) in enumerate(zip(shape, ranks)):
        stacked = torch.cat([unfold(X, k) for X in samples_t], dim=1)
        U, _, _ = torch.linalg.svd(stacked, full_matrices=False)
        A = U[:, :m]
        if scale_to_sqrt_nm:
            A = A * math.sqrt(n)
        factors.append(_to_numpy(A))
    return factors



def factor_error(A_hat: Tensor | np.ndarray, A_true: Tensor | np.ndarray) -> float:
    Ah = torch.as_tensor(A_hat, dtype=torch.float64)
    At = torch.as_tensor(A_true, dtype=torch.float64)
    C = Ah.transpose(0, 1) @ At
    U, _, Vh = torch.linalg.svd(C, full_matrices=False)
    Q = U @ Vh
    err = torch.linalg.norm(Ah @ Q - At, ord="fro") / math.sqrt(int(np.prod(At.shape)))
    return float(err.item())



def overall_factor_error(factors_hat: Sequence[Tensor | np.ndarray], factors_true: Sequence[Tensor | np.ndarray]) -> Dict[str, Any]:
    errs = [factor_error(Ah, At) for Ah, At in zip(factors_hat, factors_true)]
    return {"mode_errors": errs, "Err": float(np.mean(errs))}


# -----------------------------------------------------------------------------
# Synthetic data under the robust model
# -----------------------------------------------------------------------------


def sample_l1_tpca(
    factors: Sequence[Tensor | np.ndarray],
    lam: float,
    n_samples: int,
    rng: np.random.Generator,
    mean: Optional[Tensor | np.ndarray] = None,
    device: Optional[str | torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Sample from the isotropic Laplace scale-mixture TPCA model.

    tau_i ~ Exp(rate = lam/2), so E[tau_i] = 2/lam.
    """
    device = resolve_device(device)
    tfactors = [_to_tensor(A, device=device, dtype=dtype) for A in factors]
    shape = tuple(A.shape[0] for A in tfactors)
    ranks = tuple(A.shape[1] for A in tfactors)
    mean_tensor = torch.zeros(shape, device=device, dtype=dtype) if mean is None else _to_tensor(mean, device=device, dtype=dtype)

    samples: List[np.ndarray] = []
    taus: List[float] = []
    for _ in range(n_samples):
        Z = _randn(ranks, rng, device=device, dtype=dtype)
        tau = float(rng.exponential(scale=2.0 / lam))
        noise = _randn(shape, rng, device=device, dtype=dtype) * math.sqrt(tau)
        X = multi_mode_dot(Z, tfactors) + noise + mean_tensor
        samples.append(_to_numpy(X))
        taus.append(tau)
    return samples, np.asarray(taus, dtype=float)


# -----------------------------------------------------------------------------
# GIG moments for the latent-scale update
# -----------------------------------------------------------------------------


def _safe_bessel_ratio(nu_num: float, nu_den: float, z: np.ndarray) -> np.ndarray:
    """Return K_{nu_num}(z) / K_{nu_den}(z) using scaled Bessel K for stability."""
    z = np.maximum(np.asarray(z, dtype=float), 1e-12)
    num = kve(nu_num, z)
    den = kve(nu_den, z)
    ratio = num / np.maximum(den, 1e-300)
    return ratio



def gig_moment(nu: float, a: float, b: np.ndarray | float, order: float) -> np.ndarray:
    """E[X^order] for GIG(nu, a, b) with density x^{nu-1} exp(-(a x + b/x)/2)."""
    b = np.maximum(np.asarray(b, dtype=float), 1e-12)
    z = np.sqrt(np.maximum(a, 1e-12) * b)
    ratio = _safe_bessel_ratio(nu + order, nu, z)
    return (b / np.maximum(a, 1e-12)) ** (0.5 * order) * ratio



def gig_expected_inv_tau(nu: float, a: float, b: np.ndarray | float) -> np.ndarray:
    return gig_moment(nu, a, b, order=-1.0)



def gig_expected_tau(nu: float, a: float, b: np.ndarray | float) -> np.ndarray:
    return gig_moment(nu, a, b, order=1.0)



def tau_mode_precision(p: int, lam: float, residual_sq: np.ndarray | float) -> np.ndarray:
    residual_sq = np.maximum(np.asarray(residual_sq, dtype=float), 1e-12)
    tau_mode = (-p + np.sqrt(p * p + 4.0 * lam * residual_sq)) / (2.0 * lam)
    tau_mode = np.maximum(tau_mode, 1e-12)
    return 1.0 / tau_mode


# -----------------------------------------------------------------------------
# Robust VB-EM / ECM estimator
# -----------------------------------------------------------------------------


@dataclass
class L1TPCAVBEMResult:
    factors: List[np.ndarray]
    mean: np.ndarray
    lam: float
    weights: np.ndarray
    tau_mean: np.ndarray
    delta: np.ndarray
    z_norm_sq: np.ndarray
    latent_adjusted_score: np.ndarray
    combined_score: np.ndarray
    objective_path: List[float]
    n_iter: int
    converged: bool
    device: str


class L1TPCAVBEM:
    """Starter robust TPCA estimator based on mean-field q(z_i) q(tau_i).

    Notes
    -----
    - The E-step follows the scale-mixture model exactly.
    - The M-step is implemented as a weighted Tucker least-squares / ECM update.
    - The factor update uses posterior means of the latent cores. The posterior
      covariance S_i is used in the scale update but not fully propagated into the
      Tucker least-squares block update. This is intentional for a practical first
      research prototype.
    """

    def __init__(
        self,
        ranks: Sequence[int],
        lam_init: float = 2.0,
        update_lam: bool = True,
        tol: float = 1e-4,
        max_iter: int = 100,
        mstep_inner: int = 1,
        ridge: float = 1e-6,
        init: str = "hosvd",
        normalize_factors: bool = True,
        z_prior_precision: float = 1.0,
        lam_prior_a: float = 0.0,
        lam_prior_b: float = 0.0,
        weight_damping: float = 0.0,
        verbose: bool = False,
        device: Optional[str | torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ):
        self.ranks = tuple(int(x) for x in ranks)
        self.lam_init = float(lam_init)
        self.update_lam = bool(update_lam)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.mstep_inner = int(mstep_inner)
        self.ridge = float(ridge)
        self.init = init
        self.normalize_factors = bool(normalize_factors)
        self.z_prior_precision = float(z_prior_precision)
        self.lam_prior_a = float(lam_prior_a)
        self.lam_prior_b = float(lam_prior_b)
        self.weight_damping = float(weight_damping)
        self.verbose = bool(verbose)
        self.device = resolve_device(device)
        self.dtype = dtype

    def _normalize(self, factors: List[Tensor]) -> List[Tensor]:
        if not self.normalize_factors:
            return factors
        out: List[Tensor] = []
        for A in factors:
            target = math.sqrt(A.shape[0] * A.shape[1])
            fro = torch.linalg.norm(A, ord="fro")
            if float(fro.item()) > 0:
                A = A * (target / fro)
            out.append(A)
        return out

    def _initialize(self, samples: Sequence[Tensor]) -> Tuple[List[Tensor], Tensor]:
        if self.init == "hosvd":
            centered, mean = center_samples(samples)
            factors = [
                _to_tensor(A, device=self.device, dtype=self.dtype)
                for A in hosvd_warm_start(centered, self.ranks, device=self.device, dtype=self.dtype)
            ]
            factors = self._normalize(factors)
            return factors, mean.to(device=self.device, dtype=self.dtype)
        if self.init == "random":
            shape = tuple(samples[0].shape)
            rng = np.random.default_rng(0)
            factors = [
                _to_tensor(A, device=self.device, dtype=self.dtype)
                for A in random_factors(shape, self.ranks, rng, device=self.device, dtype=self.dtype)
            ]
            return factors, torch.stack(list(samples), dim=0).mean(dim=0)
        raise ValueError(f"unknown init={self.init!r}")

    def _e_step(
        self,
        samples: Sequence[Tensor],
        mean: Tensor,
        factors: Sequence[Tensor],
        weights: np.ndarray,
        lam: float,
    ) -> Tuple[List[Tensor], List[Tensor], np.ndarray, np.ndarray, Tensor]:
        ranks = self.ranks
        q = int(np.prod(ranks))
        p = int(np.prod(samples[0].shape))
        nu = 1.0 - 0.5 * p

        G = gram_kron(factors)
        eye_q = torch.eye(q, device=self.device, dtype=self.dtype)

        z_means: List[Tensor] = []
        z_covs: List[Tensor] = []
        deltas: List[float] = []

        beta_z = self.z_prior_precision
        for X, w in zip(samples, weights):
            centered = X - mean
            rhs_t = multi_mode_dot(centered, factors, transpose=True)
            rhs = rhs_t.reshape(-1)
            S = torch.linalg.inv(beta_z * eye_q + float(w) * G)
            m = float(w) * (S @ rhs)
            recon = multi_mode_dot(m.reshape(ranks), factors)
            resid_sq = torch.linalg.norm(centered - recon).pow(2)
            delta = float((resid_sq + torch.trace(G @ S)).item())
            z_means.append(m.reshape(ranks))
            z_covs.append(S)
            deltas.append(max(delta, 1e-12))

        delta_arr = np.asarray(deltas, dtype=float)
        tau_mean = gig_expected_tau(nu, lam, delta_arr)
        new_weights = gig_expected_inv_tau(nu, lam, delta_arr)
        new_weights = np.maximum(new_weights, 1e-10)
        tau_mean = np.maximum(tau_mean, 1e-10)
        return z_means, z_covs, new_weights, tau_mean, G

    def _update_mean(self, samples: Sequence[Tensor], z_means: Sequence[Tensor], factors: Sequence[Tensor], weights: np.ndarray) -> Tensor:
        num = torch.zeros_like(samples[0])
        den = float(np.sum(weights))
        for X, Zm, w in zip(samples, z_means, weights):
            num = num + float(w) * (X - multi_mode_dot(Zm, factors))
        return num / max(den, 1e-12)

    def _m_step_factors(self, samples: Sequence[Tensor], mean: Tensor, z_means: Sequence[Tensor], weights: np.ndarray, factors: Sequence[Tensor]) -> List[Tensor]:
        updated = [A.clone() for A in factors]
        for _ in range(self.mstep_inner):
            for k in range(len(updated)):
                n_k, m_k = updated[k].shape
                XtU = torch.zeros((n_k, m_k), device=self.device, dtype=self.dtype)
                UUt = self.ridge * torch.eye(m_k, device=self.device, dtype=self.dtype)
                for X, Zm, w in zip(samples, z_means, weights):
                    Ui = unfold(multi_mode_dot(Zm, updated, skip=k), k)
                    Xi = unfold(X - mean, k)
                    XtU = XtU + float(w) * (Xi @ Ui.transpose(0, 1))
                    UUt = UUt + float(w) * (Ui @ Ui.transpose(0, 1))
                updated[k] = XtU @ torch.linalg.inv(UUt)
            updated = self._normalize(updated)
        return updated

    def fit(
        self,
        samples: Sequence[Tensor | np.ndarray],
        init_weights: Optional[np.ndarray] = None,
        init_factors: Optional[Sequence[Tensor | np.ndarray]] = None,
        init_mean: Optional[Tensor | np.ndarray] = None,
    ) -> L1TPCAVBEMResult:
        samples_t, _ = _check_samples(samples, device=self.device, dtype=self.dtype)
        if init_factors is None:
            factors, mean = self._initialize(samples_t)
        else:
            factors = [self._normalize([_to_tensor(A, device=self.device, dtype=self.dtype) for A in init_factors])[i]
                       for i in range(len(init_factors))]
            if init_mean is None:
                _, mean = center_samples(samples_t)
                mean = mean.to(device=self.device, dtype=self.dtype)
            else:
                mean = _to_tensor(init_mean, device=self.device, dtype=self.dtype)

        p = int(np.prod(samples_t[0].shape))
        lam = float(self.lam_init)
        if init_weights is None:
            weights = np.ones(len(samples_t), dtype=float)
        else:
            weights = np.maximum(np.asarray(init_weights, dtype=float).copy(), 1e-10)
            if weights.shape[0] != len(samples_t):
                raise ValueError("init_weights must have one entry per sample")

        objective_path: List[float] = []
        converged = False
        prev = None

        for it in range(1, self.max_iter + 1):
            z_means, z_covs, new_weights, tau_mean, G = self._e_step(samples_t, mean, factors, weights, lam)
            if self.weight_damping > 0.0:
                alpha = min(max(self.weight_damping, 0.0), 1.0)
                new_weights = (1.0 - alpha) * new_weights + alpha * weights
            weights = np.maximum(new_weights, 1e-10)

            if self.update_lam:
                N = len(samples_t)
                a = self.lam_prior_a
                b = self.lam_prior_b
                lam = float(2.0 * (N + a - 1.0) / max(np.sum(tau_mean) + 2.0 * b, 1e-12))
                lam = max(lam, 1e-8)

            mean = self._update_mean(samples_t, z_means, factors, weights)
            factors = self._m_step_factors(samples_t, mean, z_means, weights, factors)

            # refresh E-step quantities for monitoring after the M-step
            z_means, z_covs, new_weights, tau_mean, G = self._e_step(samples_t, mean, factors, weights, lam)
            if self.weight_damping > 0.0:
                alpha = min(max(self.weight_damping, 0.0), 1.0)
                new_weights = (1.0 - alpha) * new_weights + alpha * weights
            weights = np.maximum(new_weights, 1e-10)

            deltas = np.zeros(len(samples_t), dtype=float)
            z_norm_sq = np.zeros(len(samples_t), dtype=float)
            log_tau = np.zeros(len(samples_t), dtype=float)
            for i, (X, Zm, S) in enumerate(zip(samples_t, z_means, z_covs)):
                centered = X - mean
                recon = multi_mode_dot(Zm, factors)
                resid_sq = torch.linalg.norm(centered - recon).pow(2)
                deltas[i] = float((resid_sq + torch.trace(G @ S)).item())
                z_norm_sq[i] = float(torch.sum(Zm.reshape(-1) ** 2).item() + torch.trace(S).item())
                # approximate E[log tau] for monitoring via mean; sufficient for a smooth objective trace.
                log_tau[i] = math.log(max(tau_mean[i], 1e-12))

            eps = 1e-12
            latent_adjusted_score = z_norm_sq / (weights + eps)
            combined_score = deltas * np.sqrt(z_norm_sq + eps) / (weights + eps)

            # A practical surrogate objective rather than the exact ELBO.
            objective = -0.5 * float(np.sum(weights * deltas + lam * tau_mean + p * log_tau))
            objective_path.append(objective)
            if self.verbose:
                print(
                    f"[L1TPCA-VBEM:{self.device}] iter={it:03d} lam={lam:.6g} "
                    f"w_mean={weights.mean():.6g} obj={objective:.6g}"
                )

            if prev is not None and abs(objective - prev) <= self.tol * (1.0 + abs(prev)):
                converged = True
                break
            prev = objective

        return L1TPCAVBEMResult(
            factors=[_to_numpy(A) for A in factors],
            mean=_to_numpy(mean),
            lam=lam,
            weights=weights.copy(),
            tau_mean=tau_mean.copy(),
            delta=deltas.copy(),
            z_norm_sq=z_norm_sq.copy(),
            latent_adjusted_score=latent_adjusted_score.copy(),
            combined_score=combined_score.copy(),
            objective_path=objective_path,
            n_iter=it,
            converged=converged,
            device=str(self.device),
        )


# -----------------------------------------------------------------------------
# Approximate robust weighted power iteration
# -----------------------------------------------------------------------------


@dataclass
class L1TPCAPowerResult:
    factors: List[np.ndarray]
    B_mats: List[np.ndarray]
    mean: np.ndarray
    lam: float
    weights: np.ndarray
    n_iter: int
    device: str


class L1TPCARobustPower:
    """Approximate weighted power iteration for the robust TPCA project.

    This is a research prototype, not a theorem-level implementation. It uses
    residual-based weights derived from the latent scale model and then applies a
    weighted version of the TPCA power update.
    """

    def __init__(
        self,
        ranks: Sequence[int],
        lam: float = 2.0,
        n_iter: int = 10,
        weight_update: str = "gig_mean",
        init: str = "random_psd",
        seed: Optional[int] = None,
        center: bool = True,
        device: Optional[str | torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ):
        self.ranks = tuple(int(x) for x in ranks)
        self.lam = float(lam)
        self.n_iter = int(n_iter)
        self.weight_update = weight_update
        self.init = init
        self.seed = seed
        self.center = bool(center)
        self.device = resolve_device(device)
        self.dtype = dtype

    def _init_B(self, shape: Sequence[int], rng: np.random.Generator) -> List[Tensor]:
        B_list: List[Tensor] = []
        for n, m in zip(shape, self.ranks):
            H = _randn((n, m), rng, device=self.device, dtype=self.dtype)
            B = H @ H.transpose(0, 1)
            B = B / torch.linalg.norm(B, ord="fro")
            B_list.append(B)
        return B_list

    @staticmethod
    def _sqrt_factor(B: Tensor, rank_cap: int) -> Tensor:
        Bs = 0.5 * (B + B.transpose(0, 1))
        evals, evecs = torch.linalg.eigh(Bs)
        idx = torch.argsort(evals, descending=True)
        evals = torch.clamp(evals[idx[:rank_cap]], min=0.0)
        evecs = evecs[:, idx[:rank_cap]]
        return evecs * torch.sqrt(evals).unsqueeze(0)

    @staticmethod
    def _top_rank_psd(B: Tensor, rank_cap: int) -> Tensor:
        Bs = 0.5 * (B + B.transpose(0, 1))
        evals, evecs = torch.linalg.eigh(Bs)
        idx = torch.argsort(evals, descending=True)[:rank_cap]
        evals = torch.clamp(evals[idx], min=0.0)
        B_rank = (evecs[:, idx] * evals.unsqueeze(0)) @ evecs[:, idx].transpose(0, 1)
        fro = torch.linalg.norm(B_rank, ord="fro")
        if float(fro.item()) == 0.0:
            return torch.eye(B.shape[0], device=B.device, dtype=B.dtype) / math.sqrt(B.shape[0])
        return B_rank / fro

    def _weights_from_residuals(self, samples: Sequence[Tensor], mean: Tensor, factors: Sequence[Tensor]) -> np.ndarray:
        p = int(np.prod(samples[0].shape))
        resid_sq = []
        for X in samples:
            centered = X - mean
            G = multi_mode_dot(centered, factors, transpose=True)
            recon = multi_mode_dot(G, factors)
            resid_sq.append(float(torch.linalg.norm(centered - recon).pow(2).item()))
        resid_sq_arr = np.asarray(resid_sq, dtype=float)
        nu = 1.0 - 0.5 * p
        if self.weight_update == "gig_mean":
            return np.maximum(gig_expected_inv_tau(nu, self.lam, resid_sq_arr), 1e-10)
        if self.weight_update == "mode":
            return np.maximum(tau_mode_precision(p, self.lam, resid_sq_arr), 1e-10)
        raise ValueError(f"unknown weight_update={self.weight_update!r}")

    def _weighted_tilde_Bk(self, samples: Sequence[Tensor], B_list: Sequence[Tensor], weights: np.ndarray, k: int) -> Tensor:
        H_list = [self._sqrt_factor(Bj, rank_cap=rj) for Bj, rj in zip(B_list, self.ranks)]
        n_k = samples[0].shape[k]
        out = torch.zeros((n_k, n_k), device=self.device, dtype=self.dtype)
        den = float(np.sum(weights))
        for X, w in zip(samples, weights):
            Y = X
            for j, H in enumerate(H_list):
                if j == k:
                    continue
                Y = mode_dot(Y, H.transpose(0, 1), j)
            Yk = unfold(Y, k)
            out = out + float(w) * (Yk @ Yk.transpose(0, 1))
        return out / max(den, 1e-12)

    def fit(self, samples: Sequence[Tensor | np.ndarray]) -> L1TPCAPowerResult:
        samples_t, shape = _check_samples(samples, device=self.device, dtype=self.dtype)
        if self.center:
            centered, mean = center_samples(samples_t)
            samples_t = [X.to(device=self.device, dtype=self.dtype) for X in centered]
            mean = mean.to(device=self.device, dtype=self.dtype)
        else:
            mean = torch.zeros(shape, device=self.device, dtype=self.dtype)

        rng = np.random.default_rng(self.seed)
        B_list = self._init_B(shape, rng)
        weights = np.ones(len(samples_t), dtype=float)

        for _ in range(self.n_iter):
            A_list = [self._sqrt_factor(B, rank_cap=r) for B, r in zip(B_list, self.ranks)]
            weights = self._weights_from_residuals(samples_t, mean, A_list)
            new_B: List[Tensor] = []
            for k in range(len(shape)):
                B_tilde = self._weighted_tilde_Bk(samples_t, B_list, weights, k)
                new_B.append(self._top_rank_psd(B_tilde, self.ranks[k]))
            B_list = new_B

        A_hat: List[np.ndarray] = []
        for k, B in enumerate(B_list):
            Bs = 0.5 * (B + B.transpose(0, 1))
            evals, evecs = torch.linalg.eigh(Bs)
            idx = torch.argsort(evals, descending=True)[: self.ranks[k]]
            evals = torch.clamp(evals[idx], min=0.0)
            evecs = evecs[:, idx]
            A_hat.append(_to_numpy(evecs * torch.sqrt(evals).unsqueeze(0)))

        return L1TPCAPowerResult(
            factors=A_hat,
            B_mats=[_to_numpy(B) for B in B_list],
            mean=_to_numpy(mean),
            lam=self.lam,
            weights=weights.copy(),
            n_iter=self.n_iter,
            device=str(self.device),
        )


# -----------------------------------------------------------------------------
# Demo helper
# -----------------------------------------------------------------------------


def demo_l1_tpca_once(
    shape: Sequence[int],
    ranks: Sequence[int],
    lam: float,
    N: int,
    rng: np.random.Generator,
    vbem_iter: int = 30,
    power_iter: int = 10,
    device: Optional[str | torch.device] = None,
) -> Dict[str, Any]:
    true_factors = random_factors(shape, ranks, rng, device=device)
    samples, taus = sample_l1_tpca(true_factors, lam=lam, n_samples=N, rng=rng, device=device)

    vbem = L1TPCAVBEM(ranks=ranks, lam_init=lam, max_iter=vbem_iter, device=device)
    fit_vbem = vbem.fit(samples)
    err_vbem = overall_factor_error(fit_vbem.factors, true_factors)

    power = L1TPCARobustPower(ranks=ranks, lam=lam, n_iter=power_iter, device=device)
    fit_power = power.fit(samples)
    err_power = overall_factor_error(fit_power.factors, true_factors)

    return {
        "shape": tuple(shape),
        "ranks": tuple(ranks),
        "lam_true": lam,
        "tau_mean_true": float(np.mean(taus)),
        "vbem": {
            "Err": err_vbem["Err"],
            "mode_errors": err_vbem["mode_errors"],
            "n_iter": fit_vbem.n_iter,
            "converged": fit_vbem.converged,
            "lam_hat": fit_vbem.lam,
            "device": fit_vbem.device,
        },
        "power": {
            "Err": err_power["Err"],
            "mode_errors": err_power["mode_errors"],
            "n_iter": fit_power.n_iter,
            "device": fit_power.device,
        },
    }


__all__ = [
    "L1TPCAVBEM",
    "L1TPCAVBEMResult",
    "L1TPCARobustPower",
    "L1TPCAPowerResult",
    "random_factors",
    "sample_l1_tpca",
    "hosvd_warm_start",
    "factor_error",
    "overall_factor_error",
    "gig_expected_inv_tau",
    "gig_expected_tau",
    "tau_mode_precision",
    "demo_l1_tpca_once",
]
