from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import numpy as np
import torch

Tensor = torch.Tensor


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


def sym(A: Tensor) -> Tensor:
    return 0.5 * (A + A.transpose(-1, -2))


# -----------------------------------------------------------------------------
# Tensor helpers
# -----------------------------------------------------------------------------


def unfold(X: Tensor, mode: int) -> Tensor:
    order = (mode,) + tuple(i for i in range(X.ndim) if i != mode)
    return X.permute(order).reshape(X.shape[mode], -1)


def fold(Xk: Tensor, shape: Sequence[int], mode: int) -> Tensor:
    shape = tuple(shape)
    order = (mode,) + tuple(i for i in range(len(shape)) if i != mode)
    inv_order = tuple(np.argsort(order).tolist())
    full = Xk.reshape([shape[mode]] + [shape[i] for i in range(len(shape)) if i != mode])
    return full.permute(inv_order)


def mode_dot(X: Tensor, A: Tensor, mode: int) -> Tensor:
    Xk = unfold(X, mode)
    Yk = A @ Xk
    new_shape = list(X.shape)
    new_shape[mode] = A.shape[0]
    return fold(Yk, new_shape, mode)


def multi_mode_dot(X: Tensor, factors: Sequence[Tensor], transpose: bool = False) -> Tensor:
    Y = X
    for k, A in enumerate(factors):
        M = A.transpose(0, 1) if transpose else A
        Y = mode_dot(Y, M, k)
    return Y


def vec_tensor(X: Tensor) -> Tensor:
    return X.reshape(-1)


def center_samples(samples: Sequence[Tensor]) -> Tuple[List[Tensor], Tensor]:
    stack = torch.stack(samples, dim=0)
    mean = stack.mean(dim=0)
    centered = [X - mean for X in samples]
    return centered, mean


def sample_covariance_from_tensors(samples: Sequence[Tensor]) -> Tensor:
    X = torch.stack([vec_tensor(s) for s in samples], dim=0)
    return (X.transpose(0, 1) @ X) / X.shape[0]


def pair_covariance(S: Tensor, shape: Sequence[int]) -> Tensor:
    """Rearrange covariance matrix to pair tensor of shape (n1^2, ..., nr^2)."""
    shape = tuple(shape)
    r = len(shape)
    T = S.reshape(*shape, *shape)
    perm = []
    for k in range(r):
        perm.extend([k, k + r])
    T = T.permute(perm)
    return T.reshape(*[n * n for n in shape])


def contract_except(T: Tensor, vectors: Sequence[Tensor], skip: int) -> Tensor:
    """Contract all modes of T except `skip` with the supplied vectors."""
    out = T
    current_modes = list(range(T.ndim))
    for original_mode in reversed(range(T.ndim)):
        if original_mode == skip:
            continue
        pos = current_modes.index(original_mode)
        out = torch.tensordot(out, vectors[original_mode], dims=([pos], [0]))
        current_modes.pop(pos)
    return out


# -----------------------------------------------------------------------------
# Synthetic TPCA data
# -----------------------------------------------------------------------------


def random_orthonormal(n: int, m: int, *, rng: np.random.Generator, device: torch.device, dtype: torch.dtype) -> Tensor:
    A = torch.from_numpy(rng.normal(size=(n, m))).to(device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(A, mode="reduced")
    return Q[:, :m]


def random_tpca_factors(
    shape: Sequence[int],
    ranks: Sequence[int],
    *,
    rng: np.random.Generator,
    device: Optional[str | torch.device] = None,
    dtype: torch.dtype = torch.float64,
    spectrum: str = "flat",
) -> List[Tensor]:
    device = resolve_device(device)
    factors: List[Tensor] = []
    for n, m in zip(shape, ranks):
        U = random_orthonormal(n, m, rng=rng, device=device, dtype=dtype)
        if spectrum == "flat":
            vals = torch.ones(m, device=device, dtype=dtype)
        elif spectrum == "linear":
            vals = torch.linspace(1.5, 0.5, m, device=device, dtype=dtype)
        else:
            raise ValueError(f"unknown spectrum={spectrum}")
        factors.append(U @ torch.diag(vals))
    return factors


def sample_tpca(
    factors: Sequence[Tensor],
    sigma2: float,
    n_samples: int,
    *,
    rng: np.random.Generator,
    mean: Optional[Tensor] = None,
    device: Optional[str | torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> List[Tensor]:
    device = resolve_device(device)
    shape = [A.shape[0] for A in factors]
    ranks = [A.shape[1] for A in factors]
    mean = torch.zeros(shape, device=device, dtype=dtype) if mean is None else _to_tensor(mean, device=device, dtype=dtype)
    out: List[Tensor] = []
    noise_sd = math.sqrt(sigma2)
    for _ in range(n_samples):
        Z = torch.from_numpy(rng.normal(size=ranks)).to(device=device, dtype=dtype)
        signal = multi_mode_dot(Z, factors)
        noise = noise_sd * torch.from_numpy(rng.normal(size=shape)).to(device=device, dtype=dtype)
        out.append(mean + signal + noise)
    return out


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def principal_subspace_error(A_true: Tensor, A_hat: Tensor) -> float:
    Qt, _ = torch.linalg.qr(A_true, mode="reduced")
    Qh, _ = torch.linalg.qr(A_hat, mode="reduced")
    Pt = Qt @ Qt.transpose(0, 1)
    Ph = Qh @ Qh.transpose(0, 1)
    num = torch.linalg.norm(Pt - Ph, ord="fro")
    den = max(torch.linalg.norm(Pt, ord="fro").item(), 1e-12)
    return float((num / den).item())


# -----------------------------------------------------------------------------
# Option A: geometric spectral/block-coordinate solver
# -----------------------------------------------------------------------------


@dataclass
class GeometricPowerResult:
    factors: List[np.ndarray]
    B_mats: List[np.ndarray]
    history: Dict[str, List[float]]
    mean: np.ndarray


class GeometricTPCAPower:
    """
    Moderate-size implementation of Option A.

    This uses the paired covariance explicitly, so it is intended for small to
    medium tensor dimensions. The algorithm can be read as block-coordinate
    ascent on the product of fixed-rank PSD sets.
    """

    def __init__(
        self,
        shape: Sequence[int],
        ranks: Sequence[int],
        *,
        max_iter: int = 30,
        tol: float = 1e-6,
        device: Optional[str | torch.device] = None,
        dtype: torch.dtype = torch.float64,
        verbose: bool = False,
    ) -> None:
        self.shape = tuple(shape)
        self.ranks = tuple(ranks)
        self.max_iter = max_iter
        self.tol = tol
        self.device = resolve_device(device)
        self.dtype = dtype
        self.verbose = verbose

    def _init_B(self, Tpair: Tensor, rng: np.random.Generator) -> List[Tensor]:
        B = []
        for n, m in zip(self.shape, self.ranks):
            M = torch.from_numpy(rng.normal(size=(n, n))).to(device=self.device, dtype=self.dtype)
            M = sym(M)
            evals, evecs = torch.linalg.eigh(M)
            idx = torch.argsort(evals, descending=True)[:m]
            lam = torch.relu(evals[idx]) + 1e-4
            U = evecs[:, idx]
            Bk = U @ torch.diag(lam) @ U.transpose(0, 1)
            Bk = Bk / torch.linalg.norm(Bk, ord="fro")
            B.append(Bk)
        return B

    def fit(self, samples: Sequence[Tensor | np.ndarray], *, rng: Optional[np.random.Generator] = None) -> GeometricPowerResult:
        rng = np.random.default_rng() if rng is None else rng
        arrs = [_to_tensor(X, device=self.device, dtype=self.dtype) for X in samples]
        centered, mean = center_samples(arrs)
        S = sample_covariance_from_tensors(centered)
        Tpair = pair_covariance(S, self.shape)

        B = self._init_B(Tpair, rng)
        obj_hist: List[float] = []
        step_hist: List[float] = []

        for it in range(self.max_iter):
            B_prev = [Bk.clone() for Bk in B]
            bvec = [vec_tensor(Bk) / torch.linalg.norm(Bk, ord="fro") for Bk in B]
            for k, (n, m) in enumerate(zip(self.shape, self.ranks)):
                raw = contract_except(Tpair, bvec, skip=k)
                M = sym(raw.reshape(n, n))
                evals, evecs = torch.linalg.eigh(M)
                idx = torch.argsort(evals, descending=True)[:m]
                lam = torch.relu(evals[idx])
                if torch.all(lam <= 1e-12):
                    lam = torch.ones_like(lam)
                Bk = evecs[:, idx] @ torch.diag(lam) @ evecs[:, idx].transpose(0, 1)
                B[k] = Bk / torch.linalg.norm(Bk, ord="fro")
                bvec[k] = vec_tensor(B[k])

            full_outer = B[0].reshape(-1)
            for k in range(1, len(B)):
                full_outer = torch.kron(full_outer, B[k].reshape(-1))
            obj = torch.dot(Tpair.reshape(-1), full_outer)
            obj_hist.append(float(obj.item()))

            step = max(float(torch.linalg.norm(a - b, ord="fro").item()) for a, b in zip(B, B_prev))
            step_hist.append(step)
            if self.verbose:
                print(f"[GeometricPower] iter={it:03d} obj={obj.item():.6e} step={step:.3e}")
            if step < self.tol:
                break

        factors = []
        for Bk, m in zip(B, self.ranks):
            evals, evecs = torch.linalg.eigh(sym(Bk))
            idx = torch.argsort(evals, descending=True)[:m]
            lam = torch.sqrt(torch.clamp(evals[idx], min=0.0))
            A = evecs[:, idx] @ torch.diag(lam)
            factors.append(_to_numpy(A))

        return GeometricPowerResult(
            factors=factors,
            B_mats=[_to_numpy(Bk) for Bk in B],
            history={"objective": obj_hist, "step": step_hist},
            mean=_to_numpy(mean),
        )


# -----------------------------------------------------------------------------
# Option B: Riemannian-style Gaussian likelihood solver
# -----------------------------------------------------------------------------


@dataclass
class RiemannianTPCAResult:
    factors: List[np.ndarray]
    B_mats: List[np.ndarray]
    sigma2: float
    mean: np.ndarray
    history: Dict[str, List[float]]


class RiemannianGaussianTPCA:
    """
    Product-manifold flavored solver for Gaussian TPCA likelihood.

    Representation:
        B_k = U_k S_k U_k^T
    where U_k lies on a Stiefel manifold and S_k is SPD. We update U_k with a
    projected gradient plus QR retraction, and optimize S_k and sigma^2 in a
    Euclidean chart. This is a practical draft, not a fully quotient-aware
    implementation.
    """

    def __init__(
        self,
        shape: Sequence[int],
        ranks: Sequence[int],
        *,
        lr_u: float = 0.2,
        lr_s: float = 0.05,
        lr_sigma: float = 0.05,
        max_iter: int = 200,
        tol: float = 1e-7,
        eps_sigma2: float = 1e-6,
        eps_spd: float = 1e-6,
        device: Optional[str | torch.device] = None,
        dtype: torch.dtype = torch.float64,
        verbose: bool = False,
    ) -> None:
        self.shape = tuple(shape)
        self.ranks = tuple(ranks)
        self.lr_u = lr_u
        self.lr_s = lr_s
        self.lr_sigma = lr_sigma
        self.max_iter = max_iter
        self.tol = tol
        self.eps_sigma2 = eps_sigma2
        self.eps_spd = eps_spd
        self.device = resolve_device(device)
        self.dtype = dtype
        self.verbose = verbose

    def _init_params(self, rng: np.random.Generator) -> Tuple[List[Tensor], List[Tensor], Tensor]:
        U_list = []
        R_list = []
        for n, m in zip(self.shape, self.ranks):
            U = random_orthonormal(n, m, rng=rng, device=self.device, dtype=self.dtype).clone().detach()
            U.requires_grad_(True)
            R = 0.05 * torch.randn((m, m), device=self.device, dtype=self.dtype)
            R = sym(R).clone().detach()
            R.requires_grad_(True)
            U_list.append(U)
            R_list.append(R)
        log_sigma2 = torch.tensor(0.0, device=self.device, dtype=self.dtype, requires_grad=True)
        return U_list, R_list, log_sigma2

    def _compose_B(self, U_list: List[Tensor], R_list: List[Tensor]) -> List[Tensor]:
        B_list = []
        eye_cache: Dict[int, Tensor] = {}
        for U, R, m in zip(U_list, R_list, self.ranks):
            eye = eye_cache.get(m)
            if eye is None:
                eye = torch.eye(m, device=self.device, dtype=self.dtype)
                eye_cache[m] = eye
            S = torch.linalg.matrix_exp(sym(R)) + self.eps_spd * eye
            B = U @ S @ U.transpose(0, 1)
            B_list.append(sym(B))
        return B_list

    def _neg_loglik(self, Xmat: Tensor, B_list: List[Tensor], sigma2: Tensor) -> Tensor:
        p = Xmat.shape[1]
        Sigma = B_list[-1]
        for k in range(len(B_list) - 2, -1, -1):
            Sigma = torch.kron(Sigma, B_list[k])
        Sigma = sym(Sigma) + sigma2 * torch.eye(p, device=self.device, dtype=self.dtype)
        sign, logabsdet = torch.linalg.slogdet(Sigma)
        if sign <= 0:
            return torch.tensor(float("inf"), device=self.device, dtype=self.dtype)
        solved = torch.linalg.solve(Sigma, Xmat.transpose(0, 1))
        quad = torch.sum(Xmat.transpose(0, 1) * solved)
        n = Xmat.shape[0]
        return 0.5 * (n * logabsdet + quad)

    def fit(self, samples: Sequence[Tensor | np.ndarray], *, rng: Optional[np.random.Generator] = None) -> RiemannianTPCAResult:
        rng = np.random.default_rng() if rng is None else rng
        arrs = [_to_tensor(X, device=self.device, dtype=self.dtype) for X in samples]
        centered, mean = center_samples(arrs)
        Xmat = torch.stack([vec_tensor(X) for X in centered], dim=0)

        U_list, R_list, log_sigma2 = self._init_params(rng)
        loss_hist: List[float] = []
        grad_hist: List[float] = []

        prev_loss = None
        for it in range(self.max_iter):
            for U in U_list:
                if U.grad is not None:
                    U.grad.zero_()
            for R in R_list:
                if R.grad is not None:
                    R.grad.zero_()
            if log_sigma2.grad is not None:
                log_sigma2.grad.zero_()

            sigma2 = torch.nn.functional.softplus(log_sigma2) + self.eps_sigma2
            B_list = self._compose_B(U_list, R_list)
            loss = self._neg_loglik(Xmat, B_list, sigma2)
            loss.backward()

            max_grad = 0.0
            with torch.no_grad():
                for idx, U in enumerate(U_list):
                    G = U.grad
                    riem_grad = G - U @ sym(U.transpose(0, 1) @ G)
                    max_grad = max(max_grad, float(torch.linalg.norm(riem_grad).item()))
                    Y = U - self.lr_u * riem_grad
                    Q, _ = torch.linalg.qr(Y, mode="reduced")
                    U.copy_(Q[:, : U.shape[1]])

                for R in R_list:
                    G = sym(R.grad)
                    max_grad = max(max_grad, float(torch.linalg.norm(G).item()))
                    R.add_(-self.lr_s * G)
                    R.copy_(sym(R))

                Gs = log_sigma2.grad
                max_grad = max(max_grad, float(torch.abs(Gs).item()))
                log_sigma2.add_(-self.lr_sigma * Gs)

            loss_val = float(loss.item())
            loss_hist.append(loss_val)
            grad_hist.append(max_grad)
            if self.verbose:
                print(f"[RiemannianGaussianTPCA] iter={it:03d} loss={loss_val:.6e} maxgrad={max_grad:.3e}")

            if prev_loss is not None and abs(prev_loss - loss_val) < self.tol and max_grad < 1e-5:
                break
            prev_loss = loss_val

        sigma2 = torch.nn.functional.softplus(log_sigma2) + self.eps_sigma2
        B_list = self._compose_B(U_list, R_list)
        factors = []
        for B, m in zip(B_list, self.ranks):
            evals, evecs = torch.linalg.eigh(sym(B))
            idx = torch.argsort(evals, descending=True)[:m]
            lam = torch.sqrt(torch.clamp(evals[idx], min=0.0))
            A = evecs[:, idx] @ torch.diag(lam)
            factors.append(_to_numpy(A))

        return RiemannianTPCAResult(
            factors=factors,
            B_mats=[_to_numpy(B) for B in B_list],
            sigma2=float(sigma2.item()),
            mean=_to_numpy(mean),
            history={"loss": loss_hist, "max_grad": grad_hist},
        )


# -----------------------------------------------------------------------------
# Convenience evaluation
# -----------------------------------------------------------------------------


def compare_factor_subspaces(true_factors: Sequence[Tensor | np.ndarray], est_factors: Sequence[Tensor | np.ndarray]) -> List[float]:
    errs: List[float] = []
    for At, Ah in zip(true_factors, est_factors):
        At_t = torch.as_tensor(At, dtype=torch.float64)
        Ah_t = torch.as_tensor(Ah, dtype=torch.float64)
        errs.append(principal_subspace_error(At_t, Ah_t))
    return errs
