from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import math
import numpy as np
import torch

from l1_tpca_torch_v2 import (
    L1TPCAVBEM,
    L1TPCARobustPower,
    overall_factor_error,
    sample_l1_tpca,
)
from tpca_reimplementation_torch import (
    TPCAEM,
    random_factors,
    sample_tpca,
)

Tensor = torch.Tensor


def _to_torch(x: np.ndarray | Tensor, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float64) -> Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


@dataclass
class ContaminationData:
    samples: List[np.ndarray]
    outlier_mask: np.ndarray
    true_factors: List[np.ndarray]
    meta: Dict[str, Any]



def generate_dataset(
    shape: Sequence[int],
    ranks: Sequence[int],
    N: int,
    scenario: str,
    sigma2: float,
    rng: np.random.Generator,
    contamination_rate: float = 0.1,
    outlier_scale: float = 8.0,
    lam: Optional[float] = None,
    slice_mode: int = 0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> ContaminationData:
    """Create datasets for robust-TPCA comparisons.

    Scenarios
    ---------
    gaussian_clean: standard Gaussian TPCA.
    laplace_clean: isotropic Laplace scale-mixture TPCA.
    tensor_outliers: Gaussian TPCA with a contaminated subset of whole-tensor outliers.
    slice_outliers: Gaussian TPCA with a contaminated subset receiving one corrupted slice.
    hybrid: Gaussian TPCA with both whole-tensor and slice contamination.
    """
    true_factors = random_factors(shape, ranks, rng, device=device, dtype=dtype)
    outlier_mask = np.zeros(N, dtype=bool)
    p = int(np.prod(shape))
    default_lam = 2.0 / max(sigma2, 1e-12)
    lam = float(default_lam if lam is None else lam)

    if scenario == "gaussian_clean":
        samples = sample_tpca(true_factors, sigma2=sigma2, n_samples=N, rng=rng, device=device, dtype=dtype)
        meta = {"scenario": scenario, "sigma2": sigma2}
        return ContaminationData(samples=samples, outlier_mask=outlier_mask, true_factors=true_factors, meta=meta)

    if scenario == "laplace_clean":
        samples, taus = sample_l1_tpca(true_factors, lam=lam, n_samples=N, rng=rng, device=device, dtype=dtype)
        meta = {"scenario": scenario, "sigma2_matched": sigma2, "lam": lam, "tau_mean": float(np.mean(taus))}
        return ContaminationData(samples=samples, outlier_mask=outlier_mask, true_factors=true_factors, meta=meta)

    # Start from a clean Gaussian TPCA dataset and contaminate a subset.
    samples = sample_tpca(true_factors, sigma2=sigma2, n_samples=N, rng=rng, device=device, dtype=dtype)
    n_bad = max(1, int(round(contamination_rate * N))) if contamination_rate > 0 else 0
    if n_bad > 0:
        bad_idx = rng.choice(N, size=n_bad, replace=False)
        outlier_mask[bad_idx] = True
    else:
        bad_idx = np.array([], dtype=int)

    scale_std = outlier_scale * math.sqrt(sigma2)
    t_samples = [_to_torch(X, device=device, dtype=dtype).clone() for X in samples]
    shape_t = tuple(shape)

    def gross_noise() -> Tensor:
        return torch.from_numpy(rng.normal(size=shape_t)).to(device=device, dtype=dtype) * scale_std

    for idx in bad_idx:
        X = t_samples[int(idx)]
        if scenario in {"tensor_outliers", "hybrid"}:
            X = X + gross_noise()
        if scenario in {"slice_outliers", "hybrid"}:
            mode = int(slice_mode) % len(shape_t)
            slicer = [slice(None)] * len(shape_t)
            slicer[mode] = int(rng.integers(0, shape_t[mode]))
            slice_shape = list(shape_t)
            del slice_shape[mode]
            corrupt_slice = torch.from_numpy(rng.normal(size=tuple(slice_shape))).to(device=device, dtype=dtype) * scale_std
            X[tuple(slicer)] = X[tuple(slicer)] + corrupt_slice
        t_samples[int(idx)] = X

    meta = {
        "scenario": scenario,
        "sigma2": sigma2,
        "contamination_rate": contamination_rate,
        "n_outliers": int(n_bad),
        "outlier_scale": outlier_scale,
        "slice_mode": slice_mode,
    }
    return ContaminationData(samples=[X.detach().cpu().numpy() for X in t_samples], outlier_mask=outlier_mask, true_factors=true_factors, meta=meta)



def _score_summary(scores: np.ndarray, outlier_mask: np.ndarray, *, low_is_outlier: bool) -> Dict[str, Any]:
    scores = np.asarray(scores, dtype=float)
    outlier_mask = np.asarray(outlier_mask, dtype=bool)
    clean_mask = ~outlier_mask
    out = {
        "all_mean": float(scores.mean()),
        "all_median": float(np.median(scores)),
        "min": float(scores.min()),
        "max": float(scores.max()),
    }
    if outlier_mask.any() and clean_mask.any():
        clean_mean = float(scores[clean_mask].mean())
        outlier_mean = float(scores[outlier_mask].mean())
        clean_median = float(np.median(scores[clean_mask]))
        outlier_median = float(np.median(scores[outlier_mask]))
        out["clean_mean"] = clean_mean
        out["outlier_mean"] = outlier_mean
        out["clean_median"] = clean_median
        out["outlier_median"] = outlier_median
        out["outlier_to_clean_mean_ratio"] = float(outlier_mean / max(clean_mean, 1e-12))
        out["outlier_to_clean_median_ratio"] = float(outlier_median / max(clean_median, 1e-12))
        ranking = np.argsort(scores) if low_is_outlier else np.argsort(-scores)
        k = int(outlier_mask.sum())
        predicted = np.zeros_like(outlier_mask, dtype=bool)
        predicted[ranking[:k]] = True
        tp = int(np.sum(predicted & outlier_mask))
        fp = int(np.sum(predicted & clean_mask))
        fn = int(np.sum((~predicted) & outlier_mask))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        tag = "low" if low_is_outlier else "high"
        out[f"topk_{tag}_score_precision"] = float(precision)
        out[f"topk_{tag}_score_recall"] = float(recall)
    return out


def _weight_summary(weights: np.ndarray, outlier_mask: np.ndarray) -> Dict[str, Any]:
    return _score_summary(weights, outlier_mask, low_is_outlier=True)



def run_one_experiment(
    shape: Sequence[int],
    ranks: Sequence[int],
    scenario: str,
    N: int,
    sigma2: float,
    seed: int,
    contamination_rate: float = 0.1,
    outlier_scale: float = 8.0,
    lam: Optional[float] = None,
    slice_mode: int = 0,
    vbem_iter: int = 40,
    power_iter: int = 10,
    gaussian_iter: int = 40,
    device: str = "auto",
    update_lam: bool = True,
    mstep_inner: int = 1,
    z_prior_precision: float = 1.0,
    lam_prior_a: float = 0.0,
    lam_prior_b: float = 0.0,
    weight_damping: float = 0.0,
    run_two_stage: bool = True,
    two_stage_blend: float = 1.0,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    data = generate_dataset(
        shape=shape,
        ranks=ranks,
        N=N,
        scenario=scenario,
        sigma2=sigma2,
        rng=rng,
        contamination_rate=contamination_rate,
        outlier_scale=outlier_scale,
        lam=lam,
        slice_mode=slice_mode,
        device=device,
    )

    lam_init = float(2.0 / max(sigma2, 1e-12) if lam is None else lam)

    robust_power = L1TPCARobustPower(
        ranks=ranks,
        lam=lam_init,
        n_iter=power_iter,
        device=device,
    )
    fit_power = robust_power.fit(data.samples)
    err_power = overall_factor_error(fit_power.factors, data.true_factors)

    robust_vbem = L1TPCAVBEM(
        ranks=ranks,
        lam_init=lam_init,
        update_lam=update_lam,
        max_iter=vbem_iter,
        mstep_inner=mstep_inner,
        z_prior_precision=z_prior_precision,
        lam_prior_a=lam_prior_a,
        lam_prior_b=lam_prior_b,
        weight_damping=weight_damping,
        verbose=False,
        device=device,
    )
    fit_vbem = robust_vbem.fit(data.samples)
    err_vbem = overall_factor_error(fit_vbem.factors, data.true_factors)

    two_stage = None
    if run_two_stage:
        blend = float(np.clip(two_stage_blend, 0.0, 1.0))
        init_weights = blend * np.asarray(fit_power.weights, dtype=float) + (1.0 - blend) * np.ones_like(fit_power.weights, dtype=float)
        vbem_stage2 = L1TPCAVBEM(
            ranks=ranks,
            lam_init=lam_init,
            update_lam=update_lam,
            max_iter=vbem_iter,
            mstep_inner=mstep_inner,
            z_prior_precision=z_prior_precision,
            lam_prior_a=lam_prior_a,
            lam_prior_b=lam_prior_b,
            weight_damping=weight_damping,
            verbose=False,
            device=device,
        )
        fit_two_stage = vbem_stage2.fit(
            data.samples,
            init_weights=init_weights,
            init_factors=fit_power.factors,
            init_mean=fit_power.mean,
        )
        err_two_stage = overall_factor_error(fit_two_stage.factors, data.true_factors)
        two_stage = {
            "Err": err_two_stage["Err"],
            "mode_errors": err_two_stage["mode_errors"],
            "lam_hat": float(fit_two_stage.lam),
            "n_iter": int(fit_two_stage.n_iter),
            "converged": bool(fit_two_stage.converged),
            "weight_summary": _score_summary(fit_two_stage.weights, data.outlier_mask, low_is_outlier=True),
            "delta_summary": _score_summary(fit_two_stage.delta, data.outlier_mask, low_is_outlier=False),
            "znorm_summary": _score_summary(fit_two_stage.z_norm_sq, data.outlier_mask, low_is_outlier=False),
            "latent_adjusted_summary": _score_summary(fit_two_stage.latent_adjusted_score, data.outlier_mask, low_is_outlier=False),
            "combined_summary": _score_summary(fit_two_stage.combined_score, data.outlier_mask, low_is_outlier=False),
        }

    gaussian = TPCAEM(
        ranks=ranks,
        sigma2_init=sigma2,
        max_iter=gaussian_iter,
        tol=1e-4,
        init="hosvd",
        update_nu=False,
        center=True,
        verbose=False,
        device=device,
    )
    fit_gauss = gaussian.fit(data.samples)
    err_gauss = overall_factor_error(fit_gauss.factors, data.true_factors)

    payload = {
        "seed": int(seed),
        "shape": tuple(shape),
        "ranks": tuple(ranks),
        "N": int(N),
        "data": data.meta,
        "robust_vbem": {
            "Err": err_vbem["Err"],
            "mode_errors": err_vbem["mode_errors"],
            "lam_hat": float(fit_vbem.lam),
            "n_iter": int(fit_vbem.n_iter),
            "converged": bool(fit_vbem.converged),
            "weight_summary": _score_summary(fit_vbem.weights, data.outlier_mask, low_is_outlier=True),
            "delta_summary": _score_summary(fit_vbem.delta, data.outlier_mask, low_is_outlier=False),
            "znorm_summary": _score_summary(fit_vbem.z_norm_sq, data.outlier_mask, low_is_outlier=False),
            "latent_adjusted_summary": _score_summary(fit_vbem.latent_adjusted_score, data.outlier_mask, low_is_outlier=False),
            "combined_summary": _score_summary(fit_vbem.combined_score, data.outlier_mask, low_is_outlier=False),
        },
        "robust_power": {
            "Err": err_power["Err"],
            "mode_errors": err_power["mode_errors"],
            "n_iter": int(fit_power.n_iter),
            "weight_summary": _score_summary(fit_power.weights, data.outlier_mask, low_is_outlier=True),
        },
        "gaussian_em": {
            "Err": err_gauss["Err"],
            "mode_errors": err_gauss["mode_errors"],
            "sigma2_hat": float(fit_gauss.sigma2),
            "n_iter": int(fit_gauss.n_iter),
            "converged": bool(fit_gauss.converged),
        },
    }
    if two_stage is not None:
        payload["two_stage_vbem"] = two_stage
    return payload


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    def collect(path: str) -> np.ndarray:
        vals = []
        for r in results:
            cur: Any = r
            for key in path.split('.'):
                cur = cur[key]
            vals.append(float(cur))
        return np.asarray(vals, dtype=float)

    summary: Dict[str, Any] = {
        "n_replications": len(results),
        "scenario": results[0]["data"]["scenario"] if results else None,
    }
    model_names = ["robust_vbem", "robust_power", "gaussian_em"]
    if results and "two_stage_vbem" in results[0]:
        model_names.insert(2, "two_stage_vbem")

    for name in model_names:
        errs = collect(f"{name}.Err")
        summary[name] = {
            "Err_mean": float(errs.mean()),
            "Err_median": float(np.median(errs)),
            "Err_std": float(errs.std(ddof=0)),
        }

    if results and results[0]["data"].get("n_outliers", 0) > 0:
        diagnostic_paths = {
            "weight": "weight_summary",
            "delta": "delta_summary",
            "znorm": "znorm_summary",
            "latent_adjusted": "latent_adjusted_summary",
            "combined": "combined_summary",
        }
        for name in [n for n in model_names if n != "gaussian_em"]:
            for tag, path in diagnostic_paths.items():
                if path not in results[0][name]:
                    continue
                ratio = collect(f"{name}.{path}.outlier_to_clean_mean_ratio")
                summary[name][f"{tag}_ratio_mean"] = float(ratio.mean())
                summary[name][f"{tag}_ratio_median"] = float(np.median(ratio))
                metric_suffix = "low_score" if tag == "weight" else "high_score"
                prec_key = f"{name}.{path}.topk_{metric_suffix}_precision"
                rec_key = f"{name}.{path}.topk_{metric_suffix}_recall"
                try:
                    prec = collect(prec_key)
                    rec = collect(rec_key)
                    summary[name][f"{tag}_topk_precision_mean"] = float(prec.mean())
                    summary[name][f"{tag}_topk_recall_mean"] = float(rec.mean())
                except Exception:
                    pass

    vbem_err = collect("robust_vbem.Err")
    gauss_err = collect("gaussian_em.Err")
    power_err = collect("robust_power.Err")
    summary["win_rates"] = {
        "vbem_better_than_gaussian": float(np.mean(vbem_err < gauss_err)),
        "power_better_than_gaussian": float(np.mean(power_err < gauss_err)),
        "vbem_better_than_power": float(np.mean(vbem_err < power_err)),
    }
    if "two_stage_vbem" in model_names:
        two_stage_err = collect("two_stage_vbem.Err")
        summary["win_rates"]["two_stage_better_than_gaussian"] = float(np.mean(two_stage_err < gauss_err))
        summary["win_rates"]["two_stage_better_than_vbem"] = float(np.mean(two_stage_err < vbem_err))
        summary["win_rates"]["two_stage_better_than_power"] = float(np.mean(two_stage_err < power_err))
    return summary


if __name__ == "__main__":
    # Minimal smoke test.
    out = run_one_experiment(
        shape=(4, 4, 4),
        ranks=(2, 2, 2),
        scenario="tensor_outliers",
        N=20,
        sigma2=1.0,
        seed=0,
        contamination_rate=0.15,
        outlier_scale=8.0,
        device="cpu",
        vbem_iter=5,
        power_iter=3,
        gaussian_iter=5,
    )
    print(json.dumps(out, indent=2))
