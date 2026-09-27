# Reproducibility Notes

> **Superseded in part (September 2026).** Current conventions are in `README.md`, `docs/STUDY_PROTOCOL.md` and `src/dp_ppnn.py`: all experiments now keep `loan_id` (12 predictors, as in the article), and the dataset file is byte-identical to the assessor's.

## Dataset and preprocessing

The supplied file contains 4,269 records and 13 columns. The paper-faithful feature count is 12 including `loan_id`. The corrected executable model removes `loan_id`, producing 11 predictors. `education` and `self_employed` are binary encoded. The train/test split is stratified 80:20 with seed 42, and standardization is fitted on the training partition only in the PyTorch audit implementation.

## Privacy mechanism

For gradient-level Laplace perturbation, per-example gradients are clipped before aggregation. The claim-aligned implementation uses L1 clipping and adds Laplace noise to the released minibatch gradient. Under replace-one adjacency the L1 sensitivity of the averaged clipped gradient is 2C/B. The PyTorch audit script uses C/(B·ε) (the add/remove-style bound); the validation suite uses 2C/(B·ε).

## Privacy accounting

A per-update epsilon is not automatically a total training epsilon. The repository therefore labels the ordinary reconstruction as `paper_protocol` and keeps a separate `strict_composed` mode. The latter explicitly allocates the declared total budget across stages and is not described as the original paper's experimental epsilon convention.

## Gaussian comparator

The Gaussian comparison is a mechanism-level reconstruction because the exact original TensorFlow Privacy invocation and accountant configuration were not available in the supplied materials.

## Result integrity

Existing result files are retained as reproducibility artifacts. New executions write to separate output directories so prior evidence is not silently overwritten.
