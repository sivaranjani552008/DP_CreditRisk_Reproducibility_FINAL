# Claim and objective status (validated, revised September 2026)

| Claim / objective | Status | Evidence |
|---|---|---|
| Dataset recoverable (4,269 rows, 2,656 / 1,613) | CONFIRMED, byte-identical to the assessor's file | scripts/verify_dataset.py |
| MLP credit-risk prediction (non-private) | CONFIRMED | results/study/summary_S0.csv |
| Published ε = 0.2 Laplace values (Table 2) | NOT REPRODUCED — withdrawn | summary_S2.csv; utility-equivalent ε ≈ 16–64 (summary_S6.csv) |
| Published DP-SGD values (Table 2) | NOT REPRODUCED — replaced | summary_S2.csv (+ RDP ε) |
| Figs 9–11 accuracies | NOT REPRODUCED — withdrawn | summary_S3_*.csv |
| Laplace > Gaussian, output stage (Alg. 3), equal ε | SUPPORTED | summary_S7.csv: significant at 9/9 ε |
| Laplace > Gaussian, input stage, per-feature ε (Eq. 2) | SUPPORTED for ε ≥ 2 | summary_S8.csv |
| Laplace > Gaussian, gradients, common clipping bound, per-update ε | SUPPORTED | summary_S3_matched.csv: 11 significant, 0 against |
| Laplace > Gaussian, Algorithm 2 | SUPPORTED | summary_S5.csv: 4 significant, 0 against |
| Laplace > Gaussian, gradients, equal total budget | SUPPORTED for 1–2 epochs only | summary_S4_matched.csv |
| Laplace > DP-SGD geometry (Gaussian + L2 clipping) | NOT SUPPORTED | summary_S3_natural.csv, summary_S4_natural.csv |
| Membership inference demonstrates privacy | NOT SUPPORTED | results/validated/validation/E5_membership_inference.csv |
| ε = 0.2 is the total privacy guarantee | NOT SUPPORTED | ε is per update; per-record total K·ε |
| Parallel Algorithm 2 | EXECUTED (declared shard rule) | summary_S5.csv |
| GDPR/HIPAA compliance | NOT ESTABLISHED BY CODE | legal/organisational question |
