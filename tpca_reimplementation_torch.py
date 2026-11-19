from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, Optional, List, Dict, Any
import math
import numpy as np
import torch

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



def kron_factors(factors: Sequence[Tensor | np.ndarray]) -> Tensor:
    out = torch.ones((1, 1), dtype=torch.float64, device=torch.as_tensor(factors[0]).device)
    for A in reversed(factors):
        A = torch.as_tensor(A)
        out = torch.kron(out, A)
    return out



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


# -----------------------------------------------------------------------------
# Data generation and metrics
# -----------------------------------------------------------------------------


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



def sample_tpca(
    factors: Sequence[Tensor | np.ndarray],
    sigma2: float,
    n_samples: int,
    rng: np.random.Generator,
    mean: Optional[Tensor | np.ndarray] = None,
    device: Optional[str | torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> List[np.ndarray]:
    device = resolve_device(device)
    tfactors = [_to_tensor(A, device=device, dtype=dtype) for A in factors]
    shape = tuple(A.shape[0] for A in tfactors)
    ranks = tuple(A.shape[1] for A in tfactors)
    mean_tensor = torch.zeros(shape, device=device, dtype=dtype) if mean is None else _to_tensor(mean, device=device, dtype=dtype)
    samples: List[np.ndarray] = []
    for _ in range(n_samples):
        Z = _randn(ranks, rng, device=device, dtype=dtype)
        noise = _randn(shape, rng, device=device, dtype=dtype) * math.sqrt(sigma2)
        X = multi_mode_dot(Z, tfactors) + noise + mean_tensor
        samples.append(_to_numpy(X))
    return samples



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
# HOSVD warm start
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# EM algorithm
# -----------------------------------------------------------------------------


@dataclass
class TPCAEMResult:
    factors: List[np.ndarray]
    sigma2: float
    mean: np.ndarray
    nu: Optional[np.ndarray]
    loglik_path: List[float]
    n_iter: int
    converged: bool
    device: str


class TPCAEM:
    def __init__(
        self,
        ranks: Sequence[int],
        tol: float = 1e-3,
        max_iter: int = 100,
        sigma2_init: Optional[float] = None,
        init: str = "hosvd",
        update_nu: bool = False,
        center: bool = True,
        verbose: bool = False,
        device: Optional[str | torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ):
        self.ranks = tuple(int(x) for x in ranks)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.sigma2_init = sigma2_init
        self.init = init
        self.update_nu = bool(update_nu)
        self.center = bool(center)
        self.verbose = bool(verbose)
        self.device = resolve_device(device)
        self.dtype = dtype

    @staticmethod
    def _compute_Wk(V: Tensor, factors: Sequence[Tensor], k: int) -> Tensor:
        ranks = [A.shape[1] for A in factors]
        r = len(ranks)
        V_tensor = V.reshape(tuple(ranks) + tuple(ranks))
        perm = []
        for j in range(r):
            perm.extend([j, j + r])
        pairV = V_tensor.permute(tuple(perm)).reshape([m * m for m in ranks])

        out = pairV
        for j in sorted([j for j in range(r) if j != k], reverse=True):
            Gj = factors[j].transpose(0, 1) @ factors[j]
            out = torch.tensordot(out, Gj.reshape(-1), dims=([j], [0]))
        if out.ndim != 1 or out.numel() != ranks[k] * ranks[k]:
            raise RuntimeError("unexpected shape when building W_k")
        return out.reshape(ranks[k], ranks[k])

    def _loglik(self, samples: Sequence[Tensor], mean: Tensor, factors: Sequence[Tensor], sigma2: float) -> float:
        M = kron_factors(factors)
        p, q = M.shape
        G = M.transpose(0, 1) @ M
        K = G + sigma2 * torch.eye(q, device=self.device, dtype=self.dtype)
        sign, logabsdet = torch.linalg.slogdet(K)
        if sign <= 0:
            return float("-inf")
        logdetSigma = (p - q) * math.log(sigma2) + float(logabsdet.item())
        Kinv = torch.linalg.inv(K)
        ll = 0.0
        for X in samples:
            x = (X - mean).reshape(-1)
            quad = (x @ x) / sigma2 - (x @ M @ Kinv @ M.transpose(0, 1) @ x) / sigma2
            quad = quad / sigma2
            ll += -0.5 * (p * math.log(2.0 * math.pi) + logdetSigma + float(quad.item()))
        return float(ll)

    def fit(self, samples: Sequence[Tensor | np.ndarray]) -> TPCAEMResult:
        samples_t, shape = _check_samples(samples, device=self.device, dtype=self.dtype)
        N = len(samples_t)
        p = int(np.prod(shape))
        ranks = self.ranks
        q = int(np.prod(ranks))

        if self.center:
            centered, mean = center_samples(samples_t)
        else:
            centered = list(samples_t)
            mean = torch.zeros(shape, device=self.device, dtype=self.dtype)

        if self.init == "hosvd":
            factors = [_to_tensor(A, device=self.device, dtype=self.dtype) for A in hosvd_warm_start(centered, ranks, device=self.device, dtype=self.dtype)]
        else:
            raise ValueError(f"unknown init={self.init!r}")

        if self.sigma2_init is None:
            total_var = float(torch.stack([(X**2).mean() for X in centered]).mean().item())
            sigma2 = max(1e-6, 0.1 * total_var)
        else:
            sigma2 = float(self.sigma2_init)

        nu = torch.zeros(ranks, device=self.device, dtype=self.dtype)
        loglik_path: List[float] = []
        converged = False

        eye_q = torch.eye(q, device=self.device, dtype=self.dtype)
        for it in range(1, self.max_iter + 1):
            M_load = kron_factors(factors)
            G = gram_kron(factors)
            M_small = G + sigma2 * eye_q
            M_small_inv = torch.linalg.inv(M_small)

            V = sigma2 * M_small_inv
            Eis: List[Tensor] = []
            latent_mean = multi_mode_dot(nu, factors)
            for X in centered:
                x = (X - latent_mean).reshape(-1)
                Ez = M_small_inv @ (M_load.transpose(0, 1) @ x)
                Eis.append(Ez.reshape(ranks))

            updated = list(factors)
            for k in range(len(shape)):
                Wk = self._compute_Wk(V, factors, k)
                XtU = torch.zeros((shape[k], ranks[k]), device=self.device, dtype=self.dtype)
                UUt = torch.zeros((ranks[k], ranks[k]), device=self.device, dtype=self.dtype)
                for X, Ei in zip(centered, Eis):
                    latent_shift = nu + Ei
                    Ui = unfold(multi_mode_dot(latent_shift, factors, skip=k), k)
                    XtU = XtU + unfold(X, k) @ Ui.transpose(0, 1)
                    UUt = UUt + Ui @ Ui.transpose(0, 1)
                XtU = XtU / N
                UUt = UUt / N
                updated[k] = XtU @ torch.linalg.inv(Wk + UUt)

            factors = updated
            M_load = kron_factors(factors)
            G = gram_kron(factors)

            if self.update_nu:
                avg_term = torch.stack([
                    X.reshape(-1) - M_load @ Ei.reshape(-1) for X, Ei in zip(centered, Eis)
                ], dim=0).mean(dim=0)
                nu_vec = torch.linalg.solve(G, M_load.transpose(0, 1) @ avg_term)
                nu = nu_vec.reshape(ranks)
            else:
                nu = torch.zeros(ranks, device=self.device, dtype=self.dtype)

            mu_tensor = multi_mode_dot(nu, factors)
            residual_ss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            for X, Ei in zip(centered, Eis):
                recon = mu_tensor + multi_mode_dot(Ei, factors)
                residual_ss = residual_ss + torch.linalg.norm(X - recon) ** 2
            tr_term = torch.trace(M_load @ V @ M_load.transpose(0, 1))
            sigma2_new = max(1e-10, float((residual_ss / (p * N) + tr_term / p).item()))
            sigma2 = sigma2_new

            ll = self._loglik(centered, torch.zeros(shape, device=self.device, dtype=self.dtype), factors, sigma2) if not self.update_nu else self._loglik(centered, mu_tensor, factors, sigma2)
            loglik_path.append(ll)
            if self.verbose:
                print(f"[EM:{self.device}] iter={it:03d} sigma2={sigma2:.6g} loglik={ll:.6g}")

            if it > 1 and abs(loglik_path[-1] - loglik_path[-2]) <= self.tol * (1.0 + abs(loglik_path[-2])):
                converged = True
                break

        total_mean = mean + multi_mode_dot(nu, factors)
        return TPCAEMResult(
            factors=[_to_numpy(A) for A in factors],
            sigma2=sigma2,
            mean=_to_numpy(total_mean),
            nu=_to_numpy(nu) if self.update_nu else None,
            loglik_path=loglik_path,
            n_iter=it,
            converged=converged,
            device=str(self.device),
        )


# -----------------------------------------------------------------------------
# Rank-1 power iteration
# -----------------------------------------------------------------------------


@dataclass
class TPCAPowerResult:
    factors: List[np.ndarray]
    B_mats: List[np.ndarray]
    sigma2: float
    omega: float
    n_iter: int
    device: str


class TPCARank1Power:
    def __init__(
        self,
        ranks: Sequence[int],
        n_iter: int = 5,
        init: str = "random_psd",
        seed: Optional[int] = None,
        device: Optional[str | torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ):
        self.ranks = tuple(int(x) for x in ranks)
        self.n_iter = int(n_iter)
        self.init = init
        self.seed = seed
        self.device = resolve_device(device)
        self.dtype = dtype

    def _init_B(self, shape: Sequence[int], rng: np.random.Generator) -> List[Tensor]:
        B_list: List[Tensor] = []
        for n, m in zip(shape, self.ranks):
            if self.init == "random_psd":
                H = _randn((n, m), rng, device=self.device, dtype=self.dtype)
                B = H @ H.transpose(0, 1)
                B = B / torch.linalg.norm(B, ord="fro")
            elif self.init == "hosvd":
                raise ValueError("hosvd init is not supported in _init_B; pass factors explicitly if needed")
            else:
                raise ValueError(f"unknown init={self.init!r}")
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

    @staticmethod
    def _tilde_Bk(samples: Sequence[Tensor], B_list: Sequence[Tensor], k: int, ranks: Sequence[int]) -> Tensor:
        H_list = [TPCARank1Power._sqrt_factor(Bj, rank_cap=rj) for Bj, rj in zip(B_list, ranks)]
        n_k = samples[0].shape[k]
        out = torch.zeros((n_k, n_k), device=samples[0].device, dtype=samples[0].dtype)
        for X in samples:
            Y = X
            for j, H in enumerate(H_list):
                if j == k:
                    continue
                Y = mode_dot(Y, H.transpose(0, 1), j)
            Yk = unfold(Y, k)
            out = out + Yk @ Yk.transpose(0, 1)
        return out / len(samples)

    @staticmethod
    def _omega_hat(samples: Sequence[Tensor], B_list: Sequence[Tensor], ranks: Sequence[int]) -> float:
        H_list = [TPCARank1Power._sqrt_factor(Bj, rank_cap=rj) for Bj, rj in zip(B_list, ranks)]
        vals = []
        for X in samples:
            Y = X
            for _, H in enumerate(H_list):
                Y = multi_mode_dot(Y, [H.transpose(0, 1) if j2 == _ else torch.eye(H.shape[0], device=Y.device, dtype=Y.dtype) for j2 in range(len(H_list))], skip=None)  # pragma: no cover
            vals.append(float(torch.linalg.norm(Y).pow(2).item()))
        return float(np.mean(vals))

    def _omega_hat_efficient(self, samples: Sequence[Tensor], B_list: Sequence[Tensor], ranks: Sequence[int]) -> float:
        H_list = [TPCARank1Power._sqrt_factor(Bj, rank_cap=rj) for Bj, rj in zip(B_list, ranks)]
        vals = []
        for X in samples:
            Y = X
            for j, H in enumerate(H_list):
                Y = mode_dot(Y, H.transpose(0, 1), j)
            vals.append(float(torch.linalg.norm(Y).pow(2).item()))
        return float(np.mean(vals))

    def fit(self, samples: Sequence[Tensor | np.ndarray], B_init: Optional[Sequence[Tensor | np.ndarray]] = None) -> TPCAPowerResult:
        samples_t, shape = _check_samples(samples, device=self.device, dtype=self.dtype)
        rng = np.random.default_rng(self.seed)
        if B_init is not None:
            B_list = [_to_tensor(B, device=self.device, dtype=self.dtype) for B in B_init]
        else:
            B_list = self._init_B(shape, rng)

        for _ in range(self.n_iter):
            new_B: List[Tensor] = []
            for k in range(len(shape)):
                B_tilde = self._tilde_Bk(samples_t, B_list, k, self.ranks)
                new_B.append(self._top_rank_psd(B_tilde, self.ranks[k]))
            B_list = new_B

        omega_hat = self._omega_hat_efficient(samples_t, B_list, self.ranks)
        scale = omega_hat ** (1.0 / len(shape))

        A_hat: List[np.ndarray] = []
        B_hat_scaled: List[np.ndarray] = []
        for k, B in enumerate(B_list):
            B_scaled = scale * B
            Bs = 0.5 * (B_scaled + B_scaled.transpose(0, 1))
            evals, evecs = torch.linalg.eigh(Bs)
            idx = torch.argsort(evals, descending=True)[: self.ranks[k]]
            evals = torch.clamp(evals[idx], min=0.0)
            evecs = evecs[:, idx]
            A_hat.append(_to_numpy(evecs * torch.sqrt(evals).unsqueeze(0)))
            B_hat_scaled.append(_to_numpy(B_scaled))

        mean_sq_norm = float(torch.stack([torch.linalg.norm(X).pow(2) for X in samples_t]).mean().item())
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
            device=str(self.device),
        )


# -----------------------------------------------------------------------------
# Section 6 helpers
# -----------------------------------------------------------------------------


def case1_em_once(
    n: int,
    m: int,
    sigma2: float,
    rng: np.random.Generator,
    n_samples: int = 15,
    tol: float = 1e-3,
    max_iter: int = 100,
    device: Optional[str | torch.device] = None,
) -> Dict[str, Any]:
    shape = (n, n, n)
    ranks = (m, m, m)
    true_factors = random_factors(shape, ranks, rng, device=device)
    samples = sample_tpca(true_factors, sigma2=sigma2, n_samples=n_samples, rng=rng, device=device)
    model = TPCAEM(
        ranks=ranks,
        tol=tol,
        max_iter=max_iter,
        sigma2_init=sigma2,
        init="hosvd",
        update_nu=False,
        center=True,
        verbose=False,
        device=device,
    )
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
        "device": fit.device,
    }



def case2_power_once(rng: np.random.Generator, n_iter: int, device: Optional[str | torch.device] = None) -> Dict[str, Any]:
    shape = (15, 15, 15)
    ranks = (3, 3, 3)
    sigma2 = 1.0
    N = 400
    true_factors = random_factors(shape, ranks, rng, device=device)
    samples = sample_tpca(true_factors, sigma2=sigma2, n_samples=N, rng=rng, device=device)
    model = TPCARank1Power(ranks=ranks, n_iter=n_iter, seed=int(rng.integers(0, 2**31 - 1)), device=device)
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
        "device": fit.device,
    }



def case3_power_once(
    shape: Sequence[int],
    ranks: Sequence[int],
    sigma2: float,
    N: int,
    rng: np.random.Generator,
    n_iter: int = 5,
    device: Optional[str | torch.device] = None,
) -> Dict[str, Any]:
    true_factors = random_factors(shape, ranks, rng, device=device)
    samples = sample_tpca(true_factors, sigma2=sigma2, n_samples=N, rng=rng, device=device)
    model = TPCARank1Power(ranks=ranks, n_iter=n_iter, seed=int(rng.integers(0, 2**31 - 1)), device=device)
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
        "device": fit.device,
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
