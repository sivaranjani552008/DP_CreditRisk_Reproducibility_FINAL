#!/usr/bin/env python3
"""
configuration_grid.py  (E6)
Full, pre-specified configuration grid comparing Laplace and Gaussian gradient
perturbation at EQUAL per-update privacy under IDENTICAL per-example L1 clipping
(sensitivity 2C/B; Gaussian std from the analytic Gaussian mechanism with
delta_step = 1e-5 / epochs). Every configuration in the grid is reported;
nothing is selected after seeing results.

Grid: B in {32, 64} x eps_update in {0.5, 1, 2, 4, 8} x epochs in {10, 30}
      x learning rate in {0.001, 0.01} x seeds {42, 43, 44}.
"""
import argparse, itertools, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import validate_claims as V

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(V.ROOT / "results" / "validated" / "configuration_grid"))
ap.add_argument("--quick", action="store_true")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
D = V.A.load_dataset(42)
X, Xt, y, yt = D["X_train"], D["X_test"], D["y_train"], D["y_test"]
grid = dict(batch=[32, 64], eps=[0.5, 1.0, 2.0, 4.0, 8.0], epochs=[10, 30], lr=[0.001, 0.01], seed=[42, 43, 44])
if a.quick:
    grid = dict(batch=[32], eps=[2.0], epochs=[1], lr=[0.001], seed=[42])
rows, t0 = [], time.time()
for b, e, ep, lr, s in itertools.product(*grid.values()):
    sig = V.analytic_gaussian_sigma(e, V.DELTA / ep)
    for mech in ["laplace", "gaussian"]:
        fn = (lambda B, e=e: ("laplace", 2 * V.C / (B * e))) if mech == "laplace" else \
             (lambda B, sig=sig: ("gaussian", sig * 2 * V.C / B))
        m, _ = V.metrics(V.train_dp(X, y, b, ep, lr, s * 1000 + b, mech, fn, clip_norm="l1"), Xt, yt)
        rows.append(dict(batch=b, eps_update=e, epochs=ep, lr=lr, seed=s, mechanism=mech,
                         per_record_eps=e * ep, **m))
    print(b, e, ep, lr, s, round(time.time() - t0), flush=True)
raw = pd.DataFrame(rows); raw.to_csv(out / "E6_grid_raw.csv", index=False)
key = ["batch", "eps_update", "epochs", "lr"]
agg = raw.groupby(key + ["mechanism"])[["accuracy", "roc_auc", "log_loss"]].mean().unstack("mechanism")
agg.columns = [f"{m}_{k}" for k, m in agg.columns]
agg = agg.reset_index()
for k in ["accuracy", "roc_auc"]:
    agg[f"diff_{k}"] = agg[f"laplace_{k}"] - agg[f"gaussian_{k}"]
agg["diff_log_loss"] = agg["gaussian_log_loss"] - agg["laplace_log_loss"]  # positive = Laplace lower loss
agg.to_csv(out / "E6_grid_summary.csv", index=False)
paired = raw.pivot_table(index=key + ["seed"], columns="mechanism", values=["accuracy", "roc_auc"]).reset_index()
summary = {
    "configurations": int(len(agg)), "paired_runs": int(len(paired)),
    "laplace_higher_auc_configs": int((agg.diff_roc_auc > 0).sum()),
    "laplace_higher_acc_configs": int((agg.diff_accuracy > 0).sum()),
    "laplace_equal_acc_configs": int((agg.diff_accuracy.abs() < 1e-12).sum()),
    "laplace_lower_loss_configs": int((agg.diff_log_loss > 0).sum()),
    "laplace_higher_auc_paired_runs": int((paired[("roc_auc", "laplace")] > paired[("roc_auc", "gaussian")]).sum()),
    "mean_auc_gain": float(agg.diff_roc_auc.mean()), "mean_acc_gain": float(agg.diff_accuracy.mean()),
    "best_laplace_acc": float(agg.laplace_accuracy.max()), "best_gaussian_acc": float(agg.gaussian_accuracy.max()),
    "runtime_seconds": time.time() - t0}
json.dump(summary, open(out / "E6_grid_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=1))
