# DP credit-risk neural network — reproducibility repository

Reproducibility materials for **Naresh, Reddi & Thamarai (2026), "Differential privacy-enabled neural
networks for secure credit risk forecasting in banking", The Journal of Supercomputing 82:516,
doi:10.1007/s11227-026-08605-3**, and for the authors' revised response to the independent
reproducibility assessment (`response/Response_to_Reproducibility_Assessment_FINAL.docx`).

This repository supports **auditable reproduction, not result matching**. No published value is used in
any computation, every configuration of the pre-specified study is reported, and results that fail to
reproduce are reported as failures.

## Main findings

| Claim | Published | Reproduced from this code |
|---|---|---|
| Non-private network | ROC-AUC 0.990 | 96.7 % accuracy, ROC-AUC 0.995 (5 seeds) |
| Laplace, ε = 0.2, 10 epochs, B = 1/32/64 (Table 2) | 92.4 / 91.8 / 90.9 % | 62.9 / 52.6 / 45.6 % with Algorithm 1 as printed — **withdrawn** |
| DP-SGD, σ = 1.1 (Table 2) | 89.7 / 87.5 / 85.2 % | 76.0 / 92.2 / 93.2 % (RDP ε 0.46 / 1.65 / 2.49) |
| Laplace beats Gaussian at equal privacy, same sensitivity | — | **Yes, in every stage**: output (Alg. 3) 9/9 ε; per-feature input 4/9 (all ε ≥ 2); gradients with a common clipping bound 11 significant, 0 against (34/36 higher mean AUC); Algorithm 2 4 significant, 0 against |
| … at equal *total* budget over training | — | Laplace better for 1–2 epochs (9 significant, 0 against); no difference at 5–10 epochs |
| Laplace beats DP-SGD geometry (Gaussian + L2 clipping) | yes | **No** — Gaussian significantly better in 26/36 (per update) and 32/32 (total budget) |
| Membership-inference AUC shows privacy | 0.512 | Uninformative — same near-chance AUC for the non-private model |

See `docs/AUDIT_FINDINGS.md` for the defects found in the previous package and how they were fixed.

## Layout

```
data/loan_approval_dataset.csv   byte-identical to the assessor's file (SHA-256 4b5cd093…4e54)
scripts/verify_dataset.py        checksum + class counts
src/dp_ppnn.py                   paper-aligned NumPy implementation: Algorithms 1-3, input stage,
                                 L1/L2 clipping, literal / per-update / total-budget calibration, ledger
experiments/study_config.json    pre-specified Laplace-vs-Gaussian grid (S0-S8)
experiments/run_study.py         runs the grid (≈28 min on 2 CPU cores)
experiments/summarize.py         paired 95 % intervals, summary tables, figures
experiments/gradient_geometry.py L1/L2 ratio of per-example gradients
results/study/                   every run (runs.csv), summaries, figures, environment
validation/validate_claims.py    PyTorch validation suite (membership inference, earlier E1-E5)
results/validated/               PyTorch validation outputs (rerun on the canonical dataset)
src/reproduce_article_claim_audit.py, src/claim_aligned_dp.py, src/claim_support_audit.py,
validation/configuration_grid.py earlier scripts, kept for traceability; superseded by src/dp_ppnn.py
response/                        response .docx, number extractor, builder, verifier
docs/AUDIT_FINDINGS.md           audit of the previous package
run_all.sh                       verify | quick | full
```

## Reproduce

```bash
pip install -r requirements.txt       # Python 3.11-3.12
./run_all.sh verify                   # checksum, unit tests, every number in the response (seconds)
./run_all.sh quick                    # smoke run of every study arm (< 1 min)
./run_all.sh full                     # full study, validation suite, summaries, response rebuild
```

Environment used for `results/study`: Python 3.11.15, NumPy 2.4.4, pandas 3.0.2, SciPy, scikit-learn 1.8.0,
dp-accounting 0.6.0, two CPU cores (see `results/study/environment.json`). The PyTorch suite used
PyTorch 2.14.0 on CPU.

## Privacy conventions

* **per update** (Lemma 1): ε is the privacy of one released gradient; per-record training budget is
  K·ε after K epochs (disjoint shuffled minibatches, basic composition).
* **total**: ε is the per-record budget for the whole run; Gaussian composed exactly, Laplace composed
  conservatively (basic pure-DP composition) — the comparison favours the Gaussian mechanism.
* Sensitivity of the mean of B clipped per-example gradients under replace-one adjacency: 2C/B.

## Notes

* `loan_id` is kept to honour the article's 12 predictors.
* GDPR/HIPAA compliance cannot be established by this code.
* Check the dataset's redistribution terms before publishing (see `DATASET_NOTICE.md`).
