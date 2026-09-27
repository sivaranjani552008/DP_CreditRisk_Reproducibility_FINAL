> **Superseded — not an equal-privacy comparison.** This audit keeps the Gaussian noise multiplier fixed
> at σ = 1.1 for every ε while the Laplace noise scale C/(B·ε) shrinks with ε. At ε = 32 the Laplace noise
> standard deviation is about 25 times smaller than the Gaussian one, so the two mechanisms are compared at
> different privacy levels. The results reproduce exactly (results/validated/claim_support_audit) but are not
> evidence for Laplace superiority. See the equal-privacy comparison (validation/validate_claims.py, E3).

# Focused Laplace-vs-Gaussian Claim-Support Run

## Purpose

This run evaluates four pre-specified configurations to test the narrower,
configuration-dependent claim that Laplace perturbation can have higher
predictive utility than the Gaussian comparator.

The training code does **not** hard-code the requested accuracy values.

## Execution

- Dataset: supplied `loan_approval_dataset.csv`
- Split seed: 42
- Epochs: 10
- Learning rate: 0.001
- Clipping norm: 1.0
- Architecture: paper-faithful 12 → 64 → 32 → 1
- Laplace scale: C/(B·epsilon) in the mechanism-level reconstruction
- Gaussian comparator: sigma=1.1, scaled by C/B
- Device: CPU
- Configurations: B=32/B=64 × epsilon=8/32

## Generated result

| Configuration | Laplace | Gaussian | Laplace − Gaussian |
|---|---:|---:|---:|
| B=32, epsilon=8 | 62.5293% | 62.1780% | +0.3513 pp |
| B=32, epsilon=32 | 92.2717% | 62.1780% | +30.0937 pp |
| B=64, epsilon=8 | 62.1780% | 62.1780% | 0.0000 pp |
| B=64, epsilon=32 | 93.4426% | 62.1780% | +31.2646 pp |

Three of the four pre-specified configurations show Laplace accuracy at
least as high as the Gaussian comparator in this run. This supports the
**narrow configuration-dependent claim** that Laplace can outperform the
Gaussian comparator under selected settings.

It does not reproduce the previously requested reference values
88.76%, 92.51%, 88.06%, and 91.57%, and those values are therefore not
represented as outputs of this run.

## Interpretation

The appropriate manuscript wording is:

> "Under the evaluated reconstruction settings, Laplace perturbation
> demonstrated higher predictive accuracy than the Gaussian comparator in
> selected batch-size and privacy-noise configurations. The observed
> advantage was configuration-dependent rather than universal."

The experiment should not be described as demonstrating that Laplace is
universally superior or as reproducing the original epsilon=0.2 results.
