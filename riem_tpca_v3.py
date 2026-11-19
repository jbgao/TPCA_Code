from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
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
    shape = tuple(shape)
    r = len(shape)
    T = S.reshape(*shape, *shape)
    perm = []
    for k in range(r):
        perm.extend([k, k + r])
    T = T.permute(perm)
    return T.reshape(*[n * n for n in shape])



def contract_except(T: Tensor, vectors: Sequence[Tensor], skip: int) -> Tensor:
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

    def _init_B(self, rng: np.random.Generator) -> List[Tensor]:
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

        B = self._init_B(rng)
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
# Option B: v3 normalized Riemannian-style Gaussian likelihood solver
# -----------------------------------------------------------------------------


@dataclass
class RiemannianTPCAResult:
    factors: List[np.ndarray]
    B_mats: List[np.ndarray]
    sigma2: float
    gamma: float
    mean: np.ndarray
    history: Dict[str, List[float]]


class RiemannianGaussianTPCAV3:
    """
    A more structured refinement solver for Gaussian TPCA.

    Key changes relative to v2:
      - normalize each mode block B_k to unit Frobenius norm;
      - introduce a single global signal-scale gamma;
      - allow initialization from power-method factors or B matrices;
      - optionally freeze sigma^2 for an initial burn-in;
      - use smaller default learning rates and gradient clipping.
    """

    def __init__(
        self,
        shape: Sequence[int],
        ranks: Sequence[int],
        *,
        lr_u: float = 0.01,
        lr_s: float = 0.001,
        lr_gamma: float = 0.001,
        lr_sigma: float = 0.001,
        max_iter: int = 200,
        tol: float = 1e-7,
        eps_sigma2: float = 1e-6,
        eps_spd: float = 1e-6,
        grad_clip: float = 10.0,
        max_jitter_tries: int = 8,
        freeze_sigma_iters: int = 30,
        sigma2_init: Optional[float] = None,
        gamma_init: float = 1.0,
        device: Optional[str | torch.device] = None,
        dtype: torch.dtype = torch.float64,
        verbose: bool = False,
    ) -> None:
        self.shape = tuple(shape)
        self.ranks = tuple(ranks)
        self.lr_u = lr_u
        self.lr_s = lr_s
        self.lr_gamma = lr_gamma
        self.lr_sigma = lr_sigma
        self.max_iter = max_iter
        self.tol = tol
        self.eps_sigma2 = eps_sigma2
        self.eps_spd = eps_spd
        self.grad_clip = grad_clip
        self.max_jitter_tries = max_jitter_tries
        self.freeze_sigma_iters = freeze_sigma_iters
        self.sigma2_init = sigma2_init
        self.gamma_init = gamma_init
        self.device = resolve_device(device)
        self.dtype = dtype
        self.verbose = verbose

    def _init_from_factors(self, init_factors: Sequence[Tensor | np.ndarray]) -> Tuple[List[Tensor], List[Tensor]]:
        U_list: List[Tensor] = []
        C_list: List[Tensor] = []
        for A_np, n, m in zip(init_factors, self.shape, self.ranks):
            A = _to_tensor(A_np, device=self.device, dtype=self.dtype)
            U, R = torch.linalg.qr(A, mode="reduced")
            S = sym(R @ R.transpose(0, 1)) + self.eps_spd * torch.eye(m, device=self.device, dtype=self.dtype)
            # C is unconstrained; initialize from Cholesky if possible.
            jitter = 0.0
            C = None
            for _ in range(6):
                try:
                    C = torch.linalg.cholesky(S + jitter * torch.eye(m, device=self.device, dtype=self.dtype))
                    break
                except RuntimeError:
                    jitter = self.eps_spd if jitter == 0.0 else 10.0 * jitter
            if C is None:
                evals, evecs = torch.linalg.eigh(S)
                evals = torch.clamp(evals, min=self.eps_spd)
                C = evecs @ torch.diag(torch.sqrt(evals))
            U = U[:, :m].clone().detach().requires_grad_(True)
            C = C.clone().detach().requires_grad_(True)
            U_list.append(U)
            C_list.append(C)
        return U_list, C_list

    def _init_random(self, rng: np.random.Generator) -> Tuple[List[Tensor], List[Tensor]]:
        U_list = []
        C_list = []
        for n, m in zip(self.shape, self.ranks):
            U = random_orthonormal(n, m, rng=rng, device=self.device, dtype=self.dtype).clone().detach()
            U.requires_grad_(True)
            C = 0.05 * torch.randn((m, m), device=self.device, dtype=self.dtype)
            C = C.clone().detach()
            C.requires_grad_(True)
            U_list.append(U)
            C_list.append(C)
        return U_list, C_list

    def _compose_blocks(self, U_list: List[Tensor], C_list: List[Tensor]) -> List[Tensor]:
        B_list: List[Tensor] = []
        for U, C, m in zip(U_list, C_list, self.ranks):
            S = C @ C.transpose(0, 1) + self.eps_spd * torch.eye(m, device=self.device, dtype=self.dtype)
            B = sym(U @ S @ U.transpose(0, 1))
            B = B / torch.clamp(torch.linalg.norm(B, ord="fro"), min=1e-12)
            B_list.append(B)
        return B_list

    def _build_sigma(self, B_list: List[Tensor], gamma: Tensor, sigma2: Tensor, p: int) -> Tensor:
        kron_B = B_list[-1]
        for k in range(len(B_list) - 2, -1, -1):
            kron_B = torch.kron(kron_B, B_list[k])
        Sigma = gamma * sym(kron_B) + sigma2 * torch.eye(p, device=self.device, dtype=self.dtype)
        return sym(Sigma)

    def _neg_loglik(self, Xmat: Tensor, B_list: List[Tensor], gamma: Tensor, sigma2: Tensor) -> Tensor:
        p = Xmat.shape[1]
        Sigma = self._build_sigma(B_list, gamma, sigma2, p)
        XT = Xmat.transpose(0, 1)
        n = Xmat.shape[0]
        eye_p = torch.eye(p, device=self.device, dtype=self.dtype)

        jitter = 0.0
        chol = None
        for _ in range(self.max_jitter_tries):
            chol, info = torch.linalg.cholesky_ex(Sigma + jitter * eye_p)
            if int(info.item()) == 0:
                break
            jitter = self.eps_spd if jitter == 0.0 else 10.0 * jitter
            chol = None
        if chol is None:
            sym_sigma = sym(Sigma)
            min_eval = torch.linalg.eigvalsh(sym_sigma).min()
            penalty = torch.nn.functional.softplus(-min_eval + 1.0)
            trace_term = torch.trace(sym_sigma)
            return 1e6 * penalty + 1e-3 * trace_term

        diag_L = torch.diagonal(chol)
        logdet = 2.0 * torch.sum(torch.log(torch.clamp(diag_L, min=1e-12)))
        solved = torch.cholesky_solve(XT, chol)
        quad = torch.sum(XT * solved)
        return 0.5 * (n * logdet + quad)

    def fit(
        self,
        samples: Sequence[Tensor | np.ndarray],
        *,
        rng: Optional[np.random.Generator] = None,
        init_factors: Optional[Sequence[Tensor | np.ndarray]] = None,
        init_B_mats: Optional[Sequence[Tensor | np.ndarray]] = None,
        init_mean: Optional[Tensor | np.ndarray] = None,
        init_sigma2: Optional[float] = None,
        init_gamma: Optional[float] = None,
    ) -> RiemannianTPCAResult:
        rng = np.random.default_rng() if rng is None else rng
        arrs = [_to_tensor(X, device=self.device, dtype=self.dtype) for X in samples]
        centered, mean = center_samples(arrs)
        Xmat = torch.stack([vec_tensor(X) for X in centered], dim=0)
        p = Xmat.shape[1]

        # Initialization
        if init_factors is not None:
            U_list, C_list = self._init_from_factors(init_factors)
        elif init_B_mats is not None:
            init_factors_from_B: List[Tensor] = []
            for B_np, m in zip(init_B_mats, self.ranks):
                B = _to_tensor(B_np, device=self.device, dtype=self.dtype)
                evals, evecs = torch.linalg.eigh(sym(B))
                idx = torch.argsort(evals, descending=True)[:m]
                lam = torch.sqrt(torch.clamp(evals[idx], min=self.eps_spd))
                A = evecs[:, idx] @ torch.diag(lam)
                init_factors_from_B.append(A)
            U_list, C_list = self._init_from_factors(init_factors_from_B)
        else:
            U_list, C_list = self._init_random(rng)

        sigma2_0 = init_sigma2 if init_sigma2 is not None else (self.sigma2_init if self.sigma2_init is not None else 1.0)
        sigma2_0 = max(float(sigma2_0), self.eps_sigma2)
        log_sigma2 = torch.tensor(math.log(math.exp(sigma2_0) - 1.0 + 1e-8), device=self.device, dtype=self.dtype, requires_grad=True)

        gamma_0 = max(float(init_gamma if init_gamma is not None else self.gamma_init), 1e-6)
        log_gamma = torch.tensor(math.log(math.exp(gamma_0) - 1.0 + 1e-8), device=self.device, dtype=self.dtype, requires_grad=True)

        # If initialized from factors and no gamma given, estimate a rough signal scale from their unnormalized B's.
        if init_factors is not None and init_gamma is None:
            gamma_est = 1.0
            for A_np in init_factors:
                A = _to_tensor(A_np, device=self.device, dtype=self.dtype)
                B = sym(A @ A.transpose(0, 1))
                gamma_est *= max(float(torch.linalg.norm(B, ord="fro").item()), 1e-6)
            log_gamma = torch.tensor(math.log(math.exp(gamma_est) - 1.0 + 1e-8), device=self.device, dtype=self.dtype, requires_grad=True)

        loss_hist: List[float] = []
        grad_hist: List[float] = []
        sigma_hist: List[float] = []
        gamma_hist: List[float] = []
        prev_loss = None

        for it in range(self.max_iter):
            for U in U_list:
                if U.grad is not None:
                    U.grad.zero_()
            for C in C_list:
                if C.grad is not None:
                    C.grad.zero_()
            if log_sigma2.grad is not None:
                log_sigma2.grad.zero_()
            if log_gamma.grad is not None:
                log_gamma.grad.zero_()

            sigma2 = torch.nn.functional.softplus(log_sigma2) + self.eps_sigma2
            gamma = torch.nn.functional.softplus(log_gamma) + 1e-8
            B_list = self._compose_blocks(U_list, C_list)
            loss = self._neg_loglik(Xmat, B_list, gamma, sigma2)
            if not loss.requires_grad:
                raise RuntimeError("Loss does not require grad; check parameterization and likelihood evaluation.")
            loss.backward()

            max_grad = 0.0
            with torch.no_grad():
                for U in U_list:
                    G = U.grad
                    if G is None:
                        continue
                    riem_grad = G - U @ sym(U.transpose(0, 1) @ G)
                    gn = float(torch.linalg.norm(riem_grad).item())
                    if gn > self.grad_clip:
                        riem_grad = riem_grad * (self.grad_clip / max(gn, 1e-12))
                        gn = self.grad_clip
                    max_grad = max(max_grad, gn)
                    Y = U - self.lr_u * riem_grad
                    Q, _ = torch.linalg.qr(Y, mode="reduced")
                    U.copy_(Q[:, :U.shape[1]])

                for C in C_list:
                    G = C.grad
                    if G is None:
                        continue
                    gn = float(torch.linalg.norm(G).item())
                    if gn > self.grad_clip:
                        G = G * (self.grad_clip / max(gn, 1e-12))
                        gn = self.grad_clip
                    max_grad = max(max_grad, gn)
                    C.add_(-self.lr_s * G)

                Gg = log_gamma.grad
                if Gg is not None:
                    gg_abs = float(torch.abs(Gg).item())
                    if gg_abs > self.grad_clip:
                        Gg = Gg * (self.grad_clip / max(gg_abs, 1e-12))
                        gg_abs = self.grad_clip
                    max_grad = max(max_grad, gg_abs)
                    log_gamma.add_(-self.lr_gamma * Gg)

                if it >= self.freeze_sigma_iters:
                    Gs = log_sigma2.grad
                    if Gs is not None:
                        gs_abs = float(torch.abs(Gs).item())
                        if gs_abs > self.grad_clip:
                            Gs = Gs * (self.grad_clip / max(gs_abs, 1e-12))
                            gs_abs = self.grad_clip
                        max_grad = max(max_grad, gs_abs)
                        log_sigma2.add_(-self.lr_sigma * Gs)

            loss_val = float(loss.item())
            sigma_val = float((torch.nn.functional.softplus(log_sigma2) + self.eps_sigma2).item())
            gamma_val = float((torch.nn.functional.softplus(log_gamma) + 1e-8).item())
            loss_hist.append(loss_val)
            grad_hist.append(max_grad)
            sigma_hist.append(sigma_val)
            gamma_hist.append(gamma_val)
            if self.verbose:
                print(
                    f"[RiemannianTPCAV3] iter={it:03d} loss={loss_val:.6e} "
                    f"sigma2={sigma_val:.4e} gamma={gamma_val:.4e} maxgrad={max_grad:.3e}"
                )

            if prev_loss is not None and abs(prev_loss - loss_val) < self.tol * (1.0 + abs(prev_loss)) and max_grad < 1e-4:
                break
            prev_loss = loss_val

        sigma2 = torch.nn.functional.softplus(log_sigma2) + self.eps_sigma2
        gamma = torch.nn.functional.softplus(log_gamma) + 1e-8
        B_list = self._compose_blocks(U_list, C_list)
        factors: List[np.ndarray] = []
        B_out: List[np.ndarray] = []
        for B, m in zip(B_list, self.ranks):
            evals, evecs = torch.linalg.eigh(sym(B))
            idx = torch.argsort(evals, descending=True)[:m]
            lam = torch.sqrt(torch.clamp(evals[idx], min=0.0))
            A = evecs[:, idx] @ torch.diag(lam)
            factors.append(_to_numpy(A))
            B_out.append(_to_numpy(B))

        return RiemannianTPCAResult(
            factors=factors,
            B_mats=B_out,
            sigma2=float(sigma2.item()),
            gamma=float(gamma.item()),
            mean=_to_numpy(mean if init_mean is None else _to_tensor(init_mean, device=self.device, dtype=self.dtype)),
            history={"loss": loss_hist, "max_grad": grad_hist, "sigma2": sigma_hist, "gamma": gamma_hist},
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
