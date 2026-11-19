# L1-TPCA starter code (PyTorch)

This is a first research prototype for the **robust TPCA / L1-TPCA** project.

## Files

- `l1_tpca_torch.py`
  - `L1TPCAVBEM`: robust TPCA with isotropic Laplace scale-mixture noise and a mean-field VB-EM / ECM routine
  - `L1TPCARobustPower`: approximate weighted power iteration using residual-based latent-scale weights
  - tensor utilities, synthetic-data generator, and evaluation helpers
- `run_l1_tpca_demo.py`
  - small CLI demo that compares
    - robust VBEM,
    - robust weighted power iteration,
    - Gaussian TPCA EM baseline.

## Model

The implemented robust model is

$$
X_i = \mathcal{M} + (A_1,\ldots,A_r)\cdot Z_i + E_i,
\qquad
Z_i \sim N(0,I),
\qquad
\mathrm{vec}(E_i)\mid \tau_i \sim N(0,\tau_i I),
\qquad
\tau_i \sim \mathrm{Exp}(\lambda/2).
$$

This is the isotropic Laplace scale-mixture version we discussed for the first stage of the project.

## Current status

### Implemented now

- exact scale-mixture E-step for
  - `q(z_i)` Gaussian,
  - `q(tau_i)` generalized inverse Gaussian;
- weighted mean update;
- weighted Tucker least-squares factor update;
- approximate robust weighted power iteration;
- synthetic data generator for the new model.

### Important caveat

The current `L1TPCAVBEM` code is a **starter research implementation**:
- the E-step follows the proposed model directly;
- the M-step uses an **ECM / weighted Tucker least-squares approximation**, based on posterior mean cores;
- the posterior covariance of `z_i` is used in the scale update, but not yet fully carried into the Tucker block update.

That makes the code suitable for prototyping and experiments, but not yet the final polished algorithm for a paper submission.

## Example

```bash
PYTHONPATH=/mnt/data python /mnt/data/run_l1_tpca_demo.py \
  --shape 6,6,6 --ranks 2,2,2 --lam 2.0 --N 50 \
  --vbem_iter 30 --power_iter 10 --gaussian_iter 30 --device auto
```

## Suggested next coding steps

1. replace the current factor M-step by a more exact expected-complete-data update;
2. add support for missing entries / masked tensors;
3. add held-out log-likelihood or ELBO diagnostics;
4. implement multiple random restarts;
5. add experiment scripts that mirror the TPCA paper figures;
6. add a robust initialization path from Gaussian TPCA.
