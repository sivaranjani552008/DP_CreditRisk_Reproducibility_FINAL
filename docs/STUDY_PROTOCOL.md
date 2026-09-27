# Settings-study protocol (Laplace vs Gaussian at equal privacy)

**Pilot (disclosed).** Before fixing the grid, 30 single-seed runs were made (seed 1, not one of the
study seeds): Algorithm 1 as printed and DP-SGD at B = 32; per-update matched/natural arms at B = 64,
ε ∈ {2, 8}, lr ∈ {0.001, 0.01}; total-budget arms at B = 64, ε ∈ {4, 16}, K ∈ {1, 2, 5, 10}. The pilot
was used only to choose ε ranges and learning rates that span chance to high utility. Pilot results are
not used anywhere.

**Grid.** `experiments/study_config.json`, sections S0–S8. Seeds 101, 202, 303, 404, 505 (S6: first three).

**Pairing.** For each configuration and seed, the two mechanisms share initialisation and minibatch
order (separate seeded streams in `src/dp_ppnn.train`); S7/S8 also share the noise stream seed.

**Primary metric.** Test ROC-AUC (the article, Sect. 6, requires it because the classes are unbalanced).
Secondary: accuracy, balanced accuracy, log-loss.

**Decision rule.** For each configuration, the paired Laplace − Gaussian difference over 5 seeds is
summarised by a two-sided 95 % t-interval. "Laplace" if the interval is above 0, "Gaussian" if below 0,
otherwise "tie" (no significant difference). No multiple-comparison correction is applied; all
configurations are reported so readers can apply one.

**Calibration.** δ = 1e-5 (article, Sect. 5); analytic Gaussian mechanism (Balle & Wang 2018).
Per-update: Gaussian at (ε, δ/K) so both mechanisms reach per-record (K·ε, δ) — Laplace with δ = 0.
Total budget: Gaussian exact composition, Laplace basic pure-DP composition (conservative).

**Stages.** S7 output (Alg. 3, S = 1, 200 noise draws per seed, non-private base model);
S8 input (features mapped to [0,1] with fixed public bounds; per-feature Eq. 2 convention and whole-record
convention; model trained on perturbed records, queried with perturbed and with clean inputs);
S3/S4/S5 gradients (Algorithms 1/2).
