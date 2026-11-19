from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import math
import numpy as np
import torch

from l1_tpca_torch import (
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



def _weight_summary(weights: np.ndarray, outlier_mask: np.ndarray) -> Dict[str, Any]:
    weights = np.asarray(weights, dtype=float)
    outlier_mask = np.asarray(outlier_mask, dtype=bool)
    clean_mask = ~outlier_mask
    out = {
        "all_mean": float(weights.mean()),
        "all_median": float(np.median(weights)),
        "min": float(weights.min()),
        "max": float(weights.max()),
    }
    if outlier_mask.any() and clean_mask.any():
        out["clean_mean"] = float(weights[clean_mask].mean())
        out["outlier_mean"] = float(weights[outlier_mask].mean())
        out["clean_median"] = float(np.median(weights[clean_mask]))
        out["outlier_median"] = float(np.median(weights[outlier_mask]))
        out["outlier_to_clean_mean_ratio"] = float(weights[outlier_mask].mean() / max(weights[clean_mask].mean(), 1e-12))
        out["outlier_to_clean_median_ratio"] = float(np.median(weights[outlier_mask]) / max(np.median(weights[clean_mask]), 1e-12))
        # Low weight should indicate outlier. This is a simple ranking diagnostic.
        ranking = np.argsort(weights)  # low weights first
        k = int(outlier_mask.sum())
        predicted = np.zeros_like(outlier_mask, dtype=bool)
        predicted[ranking[:k]] = True
        tp = int(np.sum(predicted & outlier_mask))
        fp = int(np.sum(predicted & clean_mask))
        fn = int(np.sum((~predicted) & outlier_mask))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        out["topk_low_weight_precision"] = float(precision)
        out["topk_low_weight_recall"] = float(recall)
    return out



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

    robust_vbem = L1TPCAVBEM(
        ranks=ranks,
        lam_init=lam_init,
        max_iter=vbem_iter,
        update_lam = False,  #True,
        z_prior_precision=50.0,   #Estimation-oriented regime: Use moderate z_prior_precision, for example 2, 5, 10.
                                  #Detection-oriented regime: Use large z_prior_precision, like 50.
        verbose=False,
        device=device,
    )
    fit_vbem = robust_vbem.fit(data.samples)
    err_vbem = overall_factor_error(fit_vbem.factors, data.true_factors)

    robust_power = L1TPCARobustPower(
        ranks=ranks,
        lam=lam_init,
        n_iter=power_iter,
        device=device,
    )
    fit_power = robust_power.fit(data.samples)
    err_power = overall_factor_error(fit_power.factors, data.true_factors)

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

    return {
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
            "weight_summary": _weight_summary(fit_vbem.weights, data.outlier_mask),
        },
        "robust_power": {
            "Err": err_power["Err"],
            "mode_errors": err_power["mode_errors"],
            "n_iter": int(fit_power.n_iter),
            "weight_summary": _weight_summary(fit_power.weights, data.outlier_mask),
        },
        "gaussian_em": {
            "Err": err_gauss["Err"],
            "mode_errors": err_gauss["mode_errors"],
            "sigma2_hat": float(fit_gauss.sigma2),
            "n_iter": int(fit_gauss.n_iter),
            "converged": bool(fit_gauss.converged),
        },
    }



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
    for name in ["robust_vbem", "robust_power", "gaussian_em"]:
        errs = collect(f"{name}.Err")
        summary[name] = {
            "Err_mean": float(errs.mean()),
            "Err_median": float(np.median(errs)),
            "Err_std": float(errs.std(ddof=0)),
        }

    if results and results[0]["data"].get("n_outliers", 0) > 0:
        for name in ["robust_vbem", "robust_power"]:
            ratio = collect(f"{name}.weight_summary.outlier_to_clean_mean_ratio")
            summary[name]["outlier_to_clean_weight_ratio_mean"] = float(ratio.mean())
            summary[name]["outlier_to_clean_weight_ratio_median"] = float(np.median(ratio))
            try:
                prec = collect(f"{name}.weight_summary.topk_low_weight_precision")
                rec = collect(f"{name}.weight_summary.topk_low_weight_recall")
                summary[name]["topk_low_weight_precision_mean"] = float(prec.mean())
                summary[name]["topk_low_weight_recall_mean"] = float(rec.mean())
            except KeyError:
                pass

    vbem_err = collect("robust_vbem.Err")
    gauss_err = collect("gaussian_em.Err")
    power_err = collect("robust_power.Err")
    summary["win_rates"] = {
        "vbem_better_than_gaussian": float(np.mean(vbem_err < gauss_err)),
        "power_better_than_gaussian": float(np.mean(power_err < gauss_err)),
        "vbem_better_than_power": float(np.mean(vbem_err < power_err)),
    }
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
