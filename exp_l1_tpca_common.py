from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from l1_tpca_numpy import (
    GaussianTPCAALS,
    L1TPCAVBEM,
    latent_posterior_means,
    overall_subspace_error,
    reconstruction_errors,
)


def parse_shape(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(","))


def write_json(payload: dict[str, Any], output: str | None) -> None:
    text = json.dumps(payload, indent=2)
    print(text)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


def mean_reconstruction_rmse(samples: Sequence[np.ndarray], mean: np.ndarray, factors: Sequence[np.ndarray], z_means: Sequence[np.ndarray]) -> float:
    return float(np.mean(reconstruction_errors(samples, mean, factors, z_means)))


def fit_l1_and_gaussian(
    samples: Sequence[np.ndarray],
    true_factors: Sequence[np.ndarray] | None,
    ranks: Sequence[int],
    *,
    max_iter: int,
    mstep_inner: int,
    seed: int,
    rho_init: float = 1.0,
) -> dict[str, Any]:
    l1 = L1TPCAVBEM(
        ranks,
        rho_init=rho_init,
        rho_min=0.5 * max(rho_init, 1e-3),
        rho_max=2.0 * max(rho_init, 1e-3),
        rho_damping=0.5,
        max_iter=max_iter,
        mstep_inner=mstep_inner,
        seed=seed,
        exact_second_moment=True,
    ).fit(samples)
    gauss = GaussianTPCAALS(
        ranks,
        max_iter=max_iter,
        mstep_inner=mstep_inner,
        seed=seed,
    ).fit(samples)

    out: dict[str, Any] = {
        "l1_tpca": {
            "n_iter": int(l1.n_iter),
            "converged": bool(l1.converged),
            "rho_hat": float(l1.rho),
            "rmse": mean_reconstruction_rmse(samples, l1.mean, l1.factors, l1.z_means),
            "objective_last": float(l1.objective_path[-1]) if l1.objective_path else None,
            "weights": l1.weights.astype(float).tolist(),
            "delta": l1.delta.astype(float).tolist(),
        },
        "gaussian_tpca": {
            "n_iter": int(gauss.n_iter),
            "converged": bool(gauss.converged),
            "sigma2_hat": float(gauss.sigma2),
            "rmse": mean_reconstruction_rmse(samples, gauss.mean, gauss.factors, gauss.z_means),
            "objective_last": float(gauss.objective_path[-1]) if gauss.objective_path else None,
        },
    }
    if true_factors is not None:
        out["l1_tpca"]["subspace_error"] = overall_subspace_error(l1.factors, true_factors)
        out["gaussian_tpca"]["subspace_error"] = overall_subspace_error(gauss.factors, true_factors)
    return out


def nearest_centroid_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    labels = np.asarray(sorted(set(int(y) for y in train_y if int(y) > 0)), dtype=int)
    if labels.size == 0:
        return np.zeros(test_x.shape[0], dtype=int)
    centroids = np.stack([train_x[train_y == y].mean(axis=0) for y in labels], axis=0)
    d2 = ((test_x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return labels[np.argmin(d2, axis=1)]


def classification_accuracy(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> float:
    keep_train = train_y > 0
    keep_test = test_y > 0
    if keep_train.sum() == 0 or keep_test.sum() == 0:
        return float("nan")
    pred = nearest_centroid_predict(train_x[keep_train], train_y[keep_train], test_x[keep_test])
    return float(np.mean(pred == test_y[keep_test]))


def latent_feature_matrix(samples: Sequence[np.ndarray], mean: np.ndarray, factors: Sequence[np.ndarray], *, weights: np.ndarray | None = None, noise_variance: float | None = None) -> np.ndarray:
    z = latent_posterior_means(samples, mean, factors, weights=weights, noise_variance=noise_variance)
    return np.stack([Zi.reshape(-1) for Zi in z], axis=0)
