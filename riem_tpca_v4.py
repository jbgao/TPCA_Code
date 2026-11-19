from __future__ import annotations

from typing import Optional, Sequence, List, Any, Tuple
import math
import numpy as np
import torch

from riem_tpca_v3 import (
    Tensor,
    resolve_device,
    _to_tensor,
    _to_numpy,
    sym,
    center_samples,
    vec_tensor,
    random_orthonormal,
    random_tpca_factors,
    sample_tpca,
    principal_subspace_error,
    compare_factor_subspaces,
    GeometricTPCAPower,
    GeometricPowerResult,
    RiemannianTPCAResult,
)


class RiemannianGaussianTPCAV4:
    """
    Anchored refinement version of the Gaussian TPCA likelihood solver.

    Main changes relative to v3:
      - anchor penalty toward the power-method initialization;
      - optional factor-freeze stage to refine only gamma and sigma2 first;
      - conservative defaults for learning rates;
      - same normalized block parameterization and Cholesky-stabilized likelihood.
    """

    def __init__(
        self,
        shape: Sequence[int],
        ranks: Sequence[int],
        *,
        lr_u: float = 0.002,
        lr_s: float = 5e-4,
        lr_gamma: float = 1e-3,
        lr_sigma: float = 1e-3,
        max_iter: int = 150,
        tol: float = 1e-7,
        eps_sigma2: float = 1e-6,
        eps_spd: float = 1e-6,
        grad_clip: float = 5.0,
        max_jitter_tries: int = 8,
        freeze_sigma_iters: int = 20,
        freeze_factor_iters: int = 20,
        sigma2_init: Optional[float] = None,
        gamma_init: float = 1.0,
        anchor_lambda: float = 10.0,
        anchor_decay: float = 1.0,
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
        self.freeze_factor_iters = freeze_factor_iters
        self.sigma2_init = sigma2_init
        self.gamma_init = gamma_init
        self.anchor_lambda = anchor_lambda
        self.anchor_decay = anchor_decay
        self.device = resolve_device(device)
        self.dtype = dtype
        self.verbose = verbose

    def _init_from_factors(self, init_factors: Sequence[Tensor | np.ndarray]) -> Tuple[List[Tensor], List[Tensor]]:
        U_list: List[Tensor] = []
        C_list: List[Tensor] = []
        for A_np, m in zip(init_factors, self.ranks):
            A = _to_tensor(A_np, device=self.device, dtype=self.dtype)
            U, R = torch.linalg.qr(A, mode="reduced")
            S = sym(R @ R.transpose(0, 1)) + self.eps_spd * torch.eye(m, device=self.device, dtype=self.dtype)
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
        U_list: List[Tensor] = []
        C_list: List[Tensor] = []
        for n, m in zip(self.shape, self.ranks):
            U = random_orthonormal(n, m, rng=rng, device=self.device, dtype=self.dtype).clone().detach()
            U.requires_grad_(True)
            C = (0.05 * torch.randn((m, m), device=self.device, dtype=self.dtype)).clone().detach()
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

        B_anchor = self._compose_blocks(U_list, C_list)
        B_anchor = [Bk.detach().clone() for Bk in B_anchor]

        sigma2_0 = init_sigma2 if init_sigma2 is not None else (self.sigma2_init if self.sigma2_init is not None else 1.0)
        sigma2_0 = max(float(sigma2_0), self.eps_sigma2)
        log_sigma2 = torch.tensor(math.log(math.exp(sigma2_0) - 1.0 + 1e-8), device=self.device, dtype=self.dtype, requires_grad=True)

        gamma_0 = max(float(init_gamma if init_gamma is not None else self.gamma_init), 1e-6)
        if init_factors is not None and init_gamma is None:
            gamma_est = 1.0
            for A_np in init_factors:
                A = _to_tensor(A_np, device=self.device, dtype=self.dtype)
                B = sym(A @ A.transpose(0, 1))
                gamma_est *= max(float(torch.linalg.norm(B, ord="fro").item()), 1e-6)
            gamma_0 = gamma_est
        log_gamma = torch.tensor(math.log(math.exp(gamma_0) - 1.0 + 1e-8), device=self.device, dtype=self.dtype, requires_grad=True)

        loss_hist: List[float] = []
        grad_hist: List[float] = []
        sigma_hist: List[float] = []
        gamma_hist: List[float] = []
        anchor_hist: List[float] = []
        prev_loss = None

        for it in range(self.max_iter):
            #print(it)
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
            nll = self._neg_loglik(Xmat, B_list, gamma, sigma2)
            anchor_penalty = torch.zeros((), device=self.device, dtype=self.dtype)
            lam_anchor = self.anchor_lambda / (1.0 + self.anchor_decay * it)
            if lam_anchor > 0:
                for Bk, Ba in zip(B_list, B_anchor):
                    anchor_penalty = anchor_penalty + torch.sum((Bk - Ba) ** 2)
            loss = nll + lam_anchor * anchor_penalty
            if not loss.requires_grad:
                raise RuntimeError("Loss does not require grad; check parameterization and likelihood evaluation.")
            loss.backward()

            max_grad = 0.0
            with torch.no_grad():
                factors_frozen = it < self.freeze_factor_iters
                sigma_frozen = it < self.freeze_sigma_iters

                if not factors_frozen:
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

                if not sigma_frozen:
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
            anchor_val = float(anchor_penalty.item())
            loss_hist.append(loss_val)
            grad_hist.append(max_grad)
            sigma_hist.append(sigma_val)
            gamma_hist.append(gamma_val)
            anchor_hist.append(anchor_val)
            if self.verbose:
                print(
                    f"[RiemannianTPCAV4] iter={it:03d} loss={loss_val:.6e} sigma2={sigma_val:.4e} "
                    f"gamma={gamma_val:.4e} anchor={anchor_val:.4e} maxgrad={max_grad:.3e}"
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
            history={
                "loss": loss_hist,
                "max_grad": grad_hist,
                "sigma2": sigma_hist,
                "gamma": gamma_hist,
                "anchor": anchor_hist,
            },
        )
