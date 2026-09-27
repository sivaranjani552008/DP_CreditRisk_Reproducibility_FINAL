# Audit of the DP_Final package against the published methodology and the assessment

Audit date: 26 September 2026. Scope: the DP_Final package (code, results, response), the
published article (J. Supercomputing 82:516) and the independent assessment (29 August 2026).
Severity: **Blocking** = would undermine the response if submitted as is; **Major** = a
methodological mismatch with the article; **Minor** = presentation or packaging.

| # | Severity | Finding | Resolution in this revision |
|---|---|---|---|
| 1 | Blocking | The response says the assessor's SHA-256 "has 65 hexadecimal characters". It has 64. The deposited CSV differs from the assessor's file only by a trailing comma at the end of every line (which also creates a 14th, empty column). Removing that comma reproduces the assessor's digest `4b5cd093…4e54` exactly. | `data/loan_approval_dataset.csv` replaced by the canonical bytes (parsed content unchanged); `scripts/verify_dataset.py` now checks the assessor's digest and explains the old one. The response states byte identity. |
| 2 | Blocking | The response version uploaded separately cites materials that are not in the package: `reproduce.py`, `run_all.sh`, `Makefile`, `Dockerfile`, `REPRODUCE_IN_COLAB.ipynb`, `response/Proposed_Correction.docx`, `results/validated/rerun_evidence/`, "all 16 common output files were bit-identical". | New response cites only files that exist; `run_all.sh` is now provided. |
| 3 | Major | "Algorithm 1 as written" in Table B(a) uses **L1** clipping. Algorithms 1–2 print ‖∇L_i‖₂, i.e. **L2** clipping, followed by Laplace(C/ε). | `src/dp_ppnn.py` implements the printed algorithm (L2 clip + Laplace(C/ε), mode `literal`) and reports the ε that release actually guarantees (L1 sensitivity of an L2-clipped vector is up to √d·C, d = 2,945). L1 clipping is labelled as a correction, not as "as written". |
| 4 | Major | The Laplace-superiority evidence (Table D, "36/40") counts accuracy "wins" in which Laplace sits at the majority-class rate (62.2 %) while Gaussian falls below it (46–48 %): threshold bias, not skill. Only 3 seeds, no paired intervals, no balanced accuracy, and only the per-update privacy convention. A DP-literate assessor will not accept it. | Replaced by a pre-specified, paired, 5-seed study with ROC-AUC as primary metric (the article's own imbalance argument, Sect. 6), 95 % intervals, balanced accuracy, and both privacy conventions (per update and total budget). |
| 5 | Major | No comparison at equal **total** privacy; the assessor's concern 4 is answered only by restating ε as per update. | Study S4: equal per-record total (ε, δ) with the Gaussian accounted tightly and Laplace conservatively. |
| 6 | Major | Algorithm 2 (parallel partitions) was never executed. | Implemented (per-shard clipping, per-shard noise, average of shards) and run (study S5). |
| 7 | Major | Algorithm 3 (output) and the Sect. 4.2 input stage were evaluated only at a single calibration and only with Laplace; the article claims Laplace vs Gaussian for the framework, and these single-release stages are where the mechanisms differ most. | Studies S7 (output) and S8 (input, per-feature Eq. 2 convention and record-level convention). |
| 8 | Major | Architecture ambiguity not addressed: Sect. 6 says "three fully connected layers with 12, 64 and 32 neurons" and one sigmoid output, which in Keras reads 12→[12, 64, 32]→1; the package only implements 12→64→32→1. | Both readings implemented (`arch="report"`, `arch="keras_literal"`) and reported (S0). |
| 9 | Major | Algorithm 1 step e is a plain gradient step (w ← w − η∇L), Sect. 6 says Adam; Theorem 1 adds Laplace noise to the weights without the learning rate. | `optimizer="adam"` (used, as in Sect. 6) and `optimizer="sgd"` available; the Correction states the update rule explicitly. |
| 10 | Minor | Two different non-private baselines appear in the response (96.60 % in Table A, 97.19 % in Section 6) without explanation. | One baseline table with 5 seeds for each architecture reading. |
| 11 | Minor | `.pytest_cache/` and results produced from the trailing-comma file are shipped; the manifest hashes them. | Cache removed, manifest regenerated. |

## Scientific conclusion of the audit

The article's numerical private-learning results (Table 2, Figs 9–13) cannot be defended; the
assessor is right. The comparative thesis can be defended only in a scoped form, which the new
study establishes with paired statistics:

* **Laplace dominates at equal privacy whenever both mechanisms protect the same released
  quantity with the same sensitivity** – the scalar output of Algorithm 3, the per-feature input
  release of Eq. (2), and the gradient stage when both use the same clipping set. The reason is
  analytic: for one release with L1 = L2 sensitivity Δ, Laplace needs noise s.d. √2·Δ/ε while an
  (ε, 10⁻⁵) Gaussian needs 2.3–3.4 × more over the article's ε grid (`noise_table.csv`).
* **The advantage shrinks with composition**: at equal total budget with conservative Laplace
  accounting it is significant for 1–2 epochs and not detectable at 5–10 epochs.
* **Gaussian noise with L2 clipping (DP-SGD geometry) dominates the gradient stage** of this
  2,945-parameter network, because its per-example gradients have a median L1/L2 ratio of ≈ 22
  (`results/study/gradient_geometry.json`).

This is the claim the response now makes. It is narrower than the abstract, but every part of it
is reproducible from the deposited code.
