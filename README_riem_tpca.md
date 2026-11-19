# Riemannian TPCA project draft

This folder contains a **working draft** for the geometric / Riemannian TPCA project discussed in chat.

## Files

- `riem_tpca.py`
  - `GeometricTPCAPower`: Option A code. A moderate-size implementation of the TPCA spectral method interpreted as block-coordinate ascent over fixed-rank PSD blocks `B_k`.
  - `RiemannianGaussianTPCA`: Option B code. A practical product-manifold flavored solver for the **Gaussian TPCA likelihood** using
    - Stiefel factors `U_k`
    - small SPD factors `S_k`
    - `B_k = U_k S_k U_k^T`
    - QR retraction for the Stiefel variables
    - Euclidean updates in an SPD chart for the small `S_k`
  - synthetic data generators and evaluation helpers.

- `run_riem_tpca_demo.py`
  - small demonstration script that generates Gaussian TPCA data and compares
    - the geometric power method
    - the Riemannian Gaussian likelihood solver.

## Important caveats

This is a **research prototype**, not a final optimized package.

### 1. Option A uses the explicit paired covariance tensor
That makes the code very transparent, but it is intended for **small to medium dimensions**. For larger dimensions, the efficient contractions from the TPCA paper should be substituted so that the full paired tensor is never materialized.

### 2. Option B is "Riemannian-style" rather than a full quotient-manifold implementation
The code uses a meaningful geometric parameterization,

`B_k = U_k S_k U_k^T`,

with `U_k` updated on a Stiefel manifold and `S_k` kept SPD through a matrix-exponential chart. This is already useful for the project, but it is not yet a full quotient-aware implementation with explicit horizontal lifts, exact manifold Hessians, or trust-region machinery.

### 3. The Gaussian likelihood is computed directly
The current code forms the covariance

`Sigma = kron(B_r, ..., B_1) + sigma^2 I`

explicitly. That is fine for draft experiments, but for larger problems you would want Kronecker-aware logdet / solve routines.

## Quick start

```bash
PYTHONPATH=/mnt/data python /mnt/data/run_riem_tpca_demo.py \
  --shape 5,5,5 --ranks 2,2,2 --N 40 --sigma2 1.0 --device auto
```

A smaller CPU-friendly test:

```bash
PYTHONPATH=/mnt/data python /mnt/data/run_riem_tpca_demo.py \
  --shape 4,4 --ranks 2,2 --N 20 --power_iter 5 --riem_iter 15 --device cpu
```

## Suggested next steps

1. Replace the explicit paired covariance in `GeometricTPCAPower` with the efficient contraction formula from the TPCA paper.
2. Add a proper line search or trust-region step for `RiemannianGaussianTPCA`.
3. Derive and code a robust / weighted likelihood version for the future L1-TPCA extension.
4. Add experiment scripts for multi-replication comparisons against the existing EM and power implementations.
