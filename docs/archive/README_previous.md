# Claim-aligned Differential Privacy Credit-Risk Experiment

## Dataset
- File: loan_approval_dataset(5).csv
- SHA-256: 996bfe05f4bad6bbd104ac7bb8e0329dcc0ece7c16c6956294154795d506e0ca
- Records: 4,269
- Input features after cleaning: 11
- Target: loan_status (Approved=1, Rejected=0)
- Stratified split: 80/20, random_state=42
- Train/test: 3,415 / 854
- No `loan_id` is used.
- Empty `Unnamed: 13` column is removed.

## Model
11 -> 64 -> 32 -> 1 MLP, ReLU hidden layers, sigmoid output, binary cross entropy, Adam-style gradient updates, learning rate 0.001.

## Privacy mechanisms
1. Data-level Laplace perturbation of normalized training features.
2. Per-example L1 gradient clipping at C=1.0.
3. Laplace gradient noise for the proposed mechanism.
4. Gaussian gradient noise as a comparison mechanism.
5. Laplace output perturbation.
6. No ordinary non-private `model.fit()` is performed after private training.

## Two privacy accounting modes

### paper_protocol
The epsilon value is applied separately to the data, each private gradient update, and output release, matching the mechanism-level interpretation used in the paper. **This must not be described as a composed total epsilon guarantee.**

### strict_composed
The requested total epsilon is split:
- 20% data release
- 70% training
- 10% output release

The training allocation is divided across all update steps. Gaussian noise uses a basic Gaussian-mechanism calibration with delta=1e-5 divided across update steps. This is conservative and is **not** a moments accountant.

## Important reproducibility principle
The code was written from scratch from the published methodology. It does not hard-code published accuracy values. The reported results are the actual outputs obtained from the supplied dataset.

## Main generated results
- `all_paper_protocol_results_10ep.csv`: epsilon sweep for Laplace/Gaussian, batches 1/32/64, 10 epochs.
- `epoch_curves_120.csv`: 120-epoch curves for epsilon 0.2, 0.5 and 2.0, batches 32/64.
- `strict_composed_key_results.csv`: strict-composition results for epsilon 0.2, 0.5, 2 and 8, batches 32/64.
- `non_private_baseline.csv`: non-private 11-64-32-1 MLP baseline.
- PNG files: accuracy/loss-vs-epsilon, epoch curves, ROC, PR and selected confusion matrices.

## Key limitation
The strict implementation demonstrates the privacy-accounting consequence of treating epsilon as a total budget. The paper's high epsilon=0.2 accuracy values are not reproduced by this claim-aligned implementation. They should therefore not be inserted into the results as if independently reproduced.

## Reproduction
```bash
pip install numpy pandas scikit-learn matplotlib numba
python claim_aligned_dp_from_scratch.py --data loan_approval_dataset(5).csv --out results --epochs 10 --seed 42
```
