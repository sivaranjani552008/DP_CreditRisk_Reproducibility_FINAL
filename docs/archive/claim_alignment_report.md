# Claim-aligned implementation audit

## Claims implemented
- 11-64-32-1 feed-forward neural network.
- Gradient clipping C=1.0.
- Laplace gradient perturbation.
- Gaussian comparison mechanism.
- Data perturbation stage.
- Output perturbation stage.
- Batch sizes 1, 32, 64.
- Epsilon values 0.2, 0.3, 0.4, 0.5, 1, 2, 4, 6, 8.
- No post-private ordinary training.
- Train/test preprocessing separation.
- Removal of loan_id.
- Reproducible fixed random seed.

## Claims deliberately not fabricated
The implementation does not force the published 93--96% epsilon=0.2 accuracy values. Those values were not reproduced when the proposed multi-stage mechanism was implemented directly.

The implementation also distinguishes mechanism-level epsilon from total composed privacy. A variable named epsilon alone is not treated as proof of a total privacy guarantee.

## Baseline
The non-private 11-64-32-1 MLP achieved the metrics recorded in `non_private_baseline.csv`.

## Interpretation
These experiments establish an auditable implementation of the proposed methodology. They do not establish that every numerical result in the published paper is reproducible.
