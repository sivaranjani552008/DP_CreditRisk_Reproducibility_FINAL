# Results

* `validated/` — every file regenerated from the code in this repository (logs in `validated/logs/`).
  * `validation/` — validation suite E1–E5 (`validation_results.json`, CSV tables).
  * `configuration_grid/` — E6 full pre-specified grid (raw runs, summary CSV/JSON).
  * `pytorch_audit/` — full (non-quick) run of `src/reproduce_article_claim_audit.py`.
  * `claim_aligned_sweep/` — full multi-stage sweep of `src/claim_aligned_dp.py`.
  * `claim_check/`, `claim_support_audit/` — reruns of the two earlier audits (identical to the stored versions).
* `archive_unverified/` — earlier result files and figures that the current code does not regenerate
  (older seed convention, one-epoch smoke run, or unrecorded settings). They are kept for transparency and
  are **not** used in the response.
