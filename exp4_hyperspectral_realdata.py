from __future__ import annotations

import argparse
from pathlib import Path
import struct
import zlib

import numpy as np

from exp_l1_tpca_common import classification_accuracy, latent_feature_matrix, parse_shape, write_json
from l1_tpca_numpy import GaussianTPCAALS, L1TPCAVBEM, reconstruction_errors


def _read_tag(buf: bytes, pos: int):
    raw = struct.unpack_from("<I", buf, pos)[0]
    small_type = raw & 0xFFFF
    small_size = raw >> 16
    if small_size:
        return small_type, small_size, pos + 4, pos + 8
    dtype, size = struct.unpack_from("<II", buf, pos)
    return dtype, size, pos + 8, pos + 8 + ((size + 7) // 8) * 8


def _mat5_arrays(path: Path) -> dict[str, np.ndarray]:
    type_to_dtype = {
        1: np.int8,
        2: np.uint8,
        3: np.int16,
        4: np.uint16,
        5: np.int32,
        6: np.uint32,
        7: np.float32,
        9: np.float64,
    }
    data = path.read_bytes()
    pos = 128
    arrays: dict[str, np.ndarray] = {}
    while pos + 8 <= len(data):
        dtype, size, payload_pos, next_pos = _read_tag(data, pos)
        payload = data[payload_pos : payload_pos + size]
        pos = next_pos
        if dtype == 15:
            payload = zlib.decompress(payload)
            inner_type, inner_size, inner_pos, _ = _read_tag(payload, 0)
            if inner_type != 14:
                continue
            payload = payload[inner_pos : inner_pos + inner_size]
            dtype = 14
        if dtype != 14:
            continue
        p = 0
        _, _, _, p_next = _read_tag(payload, p)
        p = p_next
        _, dim_size, dim_pos, p_next = _read_tag(payload, p)
        dims = tuple(int(x) for x in np.frombuffer(payload[dim_pos : dim_pos + dim_size], dtype="<i4"))
        p = p_next
        _, name_size, name_pos, p_next = _read_tag(payload, p)
        name = payload[name_pos : name_pos + name_size].decode("latin1")
        p = p_next
        real_type, real_size, real_pos, _ = _read_tag(payload, p)
        if real_type not in type_to_dtype:
            continue
        arr = np.frombuffer(payload[real_pos : real_pos + real_size], dtype=np.dtype(type_to_dtype[real_type]).newbyteorder("<"))
        arrays[name] = np.array(arr.reshape(dims, order="F"))
    return arrays


def load_mat_array(path: Path, preferred_keys: tuple[str, ...]) -> np.ndarray:
    try:
        from scipy.io import loadmat  # type: ignore
        data = loadmat(path)
    except Exception:
        data = _mat5_arrays(path)
    for key in preferred_keys:
        if key in data:
            return np.asarray(data[key])
    candidates = [v for k, v in data.items() if not k.startswith("__") and isinstance(v, np.ndarray)]
    if not candidates:
        raise ValueError(f"No array found in {path}")
    return np.asarray(candidates[0])


def extract_patches(cube: np.ndarray, gt: np.ndarray | None, *, patch_size: int, stride: int, max_patches: int, seed: int):
    rng = np.random.default_rng(seed)
    h, w, _ = cube.shape
    samples = []
    labels = []
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = cube[i : i + patch_size, j : j + patch_size, :]
            samples.append(patch.astype(float))
            if gt is None:
                labels.append(0)
            else:
                lab = gt[i : i + patch_size, j : j + patch_size].reshape(-1)
                lab = lab[lab > 0]
                labels.append(int(np.bincount(lab.astype(int)).argmax()) if lab.size else 0)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    idx = idx[: min(max_patches, len(idx))]
    return [samples[i] for i in idx], np.asarray([labels[i] for i in idx], dtype=int)


def standardize_cube(cube: np.ndarray) -> np.ndarray:
    cube = np.asarray(cube, dtype=float)
    flat = cube.reshape(-1, cube.shape[2])
    mu = flat.mean(axis=0)
    sd = flat.std(axis=0)
    return (cube - mu) / np.maximum(sd, 1e-8)


def corrupt_test_patches(samples, rng, *, rate: float, scale: float):
    out = [X.copy() for X in samples]
    n_bad = int(round(rate * len(out)))
    mask = np.zeros(len(out), dtype=bool)
    if n_bad <= 0:
        return out, mask
    for i in rng.choice(len(out), size=n_bad, replace=False):
        mask[i] = True
        band = int(rng.integers(0, out[i].shape[2]))
        out[i][:, :, band] += rng.normal(scale=scale, size=out[i][:, :, band].shape)
    return out, mask


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Experiment 4: hyperspectral patch experiment.")
    ap.add_argument("--dataset", choices=["indian_pines", "salinas"], default="indian_pines")
    ap.add_argument("--data_dir", type=str, default="data/hyperspectral")
    ap.add_argument("--ranks", type=parse_shape, default="3,3,4")
    ap.add_argument("--patch_size", type=int, default=9)
    ap.add_argument("--stride", type=int, default=9)
    ap.add_argument("--max_patches", type=int, default=300)
    ap.add_argument("--train_fraction", type=float, default=0.7)
    ap.add_argument("--test_corruption_rate", type=float, default=0.1)
    ap.add_argument("--test_corruption_scale", type=float, default=8.0)
    ap.add_argument("--max_iter", type=int, default=30)
    ap.add_argument("--mstep_inner", type=int, default=1)
    ap.add_argument("--seed", type=int, default=400)
    ap.add_argument("--output", type=str, default="results/exp4_hyperspectral_realdata.json")
    args = ap.parse_args()

    root = Path(args.data_dir)
    if args.dataset == "indian_pines":
        data_path = root / "Indian_pines_corrected.mat"
        gt_path = root / "Indian_pines_gt.mat"
        cube = load_mat_array(data_path, ("indian_pines_corrected", "indian_pines"))
        gt = load_mat_array(gt_path, ("indian_pines_gt",))
    else:
        data_path = root / "Salinas_corrected.mat"
        gt_path = root / "Salinas_gt.mat"
        cube = load_mat_array(data_path, ("salinas_corrected", "salinas"))
        gt = load_mat_array(gt_path, ("salinas_gt",))

    cube = standardize_cube(cube)
    samples, labels = extract_patches(
        cube,
        gt,
        patch_size=args.patch_size,
        stride=args.stride,
        max_patches=args.max_patches,
        seed=args.seed,
    )
    rng = np.random.default_rng(args.seed)
    n_train = int(round(args.train_fraction * len(samples)))
    train_samples, test_samples = samples[:n_train], samples[n_train:]
    train_y, test_y = labels[:n_train], labels[n_train:]
    test_samples_corrupt, corrupt_mask = corrupt_test_patches(
        test_samples,
        rng,
        rate=args.test_corruption_rate,
        scale=args.test_corruption_scale,
    )

    l1 = L1TPCAVBEM(args.ranks, max_iter=args.max_iter, mstep_inner=args.mstep_inner, seed=args.seed).fit(train_samples)
    gauss = GaussianTPCAALS(args.ranks, max_iter=args.max_iter, mstep_inner=args.mstep_inner, seed=args.seed).fit(train_samples)

    l1_train_x = np.stack([z.reshape(-1) for z in l1.z_means], axis=0)
    l1_test_x = latent_feature_matrix(test_samples_corrupt, l1.mean, l1.factors)
    g_train_x = np.stack([z.reshape(-1) for z in gauss.z_means], axis=0)
    g_test_x = latent_feature_matrix(test_samples_corrupt, gauss.mean, gauss.factors, noise_variance=gauss.sigma2)

    l1_test_z = latent_feature_matrix(test_samples_corrupt, l1.mean, l1.factors)
    g_test_z = latent_feature_matrix(test_samples_corrupt, gauss.mean, gauss.factors, noise_variance=gauss.sigma2)

    payload = {
        "config": vars(args),
        "data_shape": list(cube.shape),
        "n_train": len(train_samples),
        "n_test": len(test_samples),
        "test_corrupt_count": int(corrupt_mask.sum()),
        "l1_tpca": {
            "rho_hat": float(l1.rho),
            "train_rmse": float(np.mean(reconstruction_errors(train_samples, l1.mean, l1.factors, l1.z_means))),
            "test_rmse": float(np.mean(reconstruction_errors(test_samples_corrupt, l1.mean, l1.factors, [z.reshape(args.ranks) for z in l1_test_z]))),
            "nearest_centroid_accuracy": classification_accuracy(l1_train_x, train_y, l1_test_x, test_y),
        },
        "gaussian_tpca": {
            "sigma2_hat": float(gauss.sigma2),
            "train_rmse": float(np.mean(reconstruction_errors(train_samples, gauss.mean, gauss.factors, gauss.z_means))),
            "test_rmse": float(np.mean(reconstruction_errors(test_samples_corrupt, gauss.mean, gauss.factors, [z.reshape(args.ranks) for z in g_test_z]))),
            "nearest_centroid_accuracy": classification_accuracy(g_train_x, train_y, g_test_x, test_y),
        },
    }
    write_json(payload, args.output)
