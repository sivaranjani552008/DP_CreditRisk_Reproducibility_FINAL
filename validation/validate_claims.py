#!/usr/bin/env python3
"""
validate_claims.py
Independent-concern validation suite for Naresh, Reddi & Thamarai (2026),
J. Supercomputing 82:516.  Answers the six clarification requests of the
independent reproducibility assessment with executed, fixed-seed experiments.

Reuses the repository's paper-faithful PyTorch reconstruction
(src/reproduce_article_claim_audit.py: data loading, 12->64->32->1 MLP,
per-example L1 clipping, Adam) and adds, without changing that code:
  E1  non-private baselines (12-feature paper-faithful and 11-feature variant)
  E2  Table 2 reconstruction at epsilon=0.2, 10 epochs, B=1/32/64:
        (a) literal Algorithm 1 noise Laplace(C/eps) on the averaged clipped gradient
        (b) sensitivity-calibrated noise Laplace(2C/(B*eps)) (replace-one adjacency)
        (c) DP-SGD comparator: L2 clipping C=1, sigma=1.1, lr=0.01 (paper's DP-SGD setting)
  E3  equal-privacy Laplace vs Gaussian: same clipping, same sensitivity 2C/B,
      same per-update (eps, delta_step); Gaussian std from the analytic Gaussian
      mechanism (Balle & Wang 2018); 3 seeds
  E4  privacy accounting for every configuration (per-update, per-record
      composition over epochs, naive all-step composition, RDP for DP-SGD)
  E5  membership inference: paper protocol and a held-out, class-balanced attack
No published value is used for any computation or parameter choice.
"""
import argparse, json, math, platform, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import reproduce_article_claim_audit as A  # repository implementation

C = 1.0
DELTA = 1e-5


# ------------------------------------------------------------------ helpers
def analytic_gaussian_sigma(eps, delta, sens=1.0):
    """Smallest sigma such that N(0, sigma^2) with L2 sensitivity `sens` is (eps, delta)-DP
    (Balle & Wang 2018, Theorem 8), found by bisection."""
    def delta_of(s):
        a = sens / (2 * s); b = eps * s / sens
        return norm.cdf(a - b) - math.exp(eps) * norm.cdf(-a - b)
    lo, hi = 1e-6, 1e6
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if delta_of(mid) > delta: lo = mid
        else: hi = mid
    return hi


def train_dp(X, y, batch, epochs, lr, seed, mech, noise_std_fn, clip_norm="l1"):
    """Same loop as repository train_private (per-example clipping, mean, noise, Adam),
    with the noise scale supplied explicitly so that its calibration is auditable.
    noise_std_fn(batch_len) -> ('laplace', b) or ('gaussian', std)."""
    A.set_seed(seed)
    model = A.CreditMLP(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    g = torch.Generator().manual_seed(seed + 31)
    for ep in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        for st in range(0, len(X), batch):
            idx = perm[st:st + batch]
            grads = A.clipped_batch_gradient(model, Xt[idx], yt[idx], C, norm_type=clip_norm)
            kind, s = noise_std_fn(len(idx))
            for gr in grads:
                if kind == "laplace":
                    gr.add_(torch.distributions.Laplace(0.0, s).sample(gr.shape))
                else:
                    gr.add_(torch.randn_like(gr) * s)
            opt.zero_grad(set_to_none=True)
            for p, gr in zip(model.parameters(), grads): p.grad = gr
            opt.step()
    return model


def metrics(model, X, y):
    p = A.predict(model, X, torch.device("cpu"))
    m = A.metric_bundle(np.asarray(y, dtype=np.int64), p)
    return {k: float(m[k]) for k in ["accuracy", "roc_auc", "log_loss", "f1"]}, p


def nonprivate(X, y, batch, epochs, seed):
    model, _ = A.train_nonprivate(X, y, batch_size=batch, epochs=epochs, learning_rate=0.001,
                                  seed=seed, device=torch.device("cpu"))
    return model


def mia(train_p, test_p, y_tr, y_te, seed):
    """Paper protocol (fit/evaluate on same rows) and a held-out, class-balanced attack
    on the confidence assigned to the true label."""
    conf_tr = np.where(y_tr == 1, train_p, 1 - train_p)
    conf_te = np.where(y_te == 1, test_p, 1 - test_p)
    Xa = np.concatenate([train_p, test_p]).reshape(-1, 1)
    ya = np.concatenate([np.ones(len(train_p)), np.zeros(len(test_p))])
    lr = LogisticRegression().fit(Xa, ya)
    paper_auc = roc_auc_score(ya, lr.predict_proba(Xa)[:, 1])
    rng = np.random.default_rng(seed)
    mem = rng.choice(len(conf_tr), len(conf_te), replace=False)
    Xb = np.concatenate([conf_tr[mem], conf_te]).reshape(-1, 1)
    yb = np.concatenate([np.ones(len(mem)), np.zeros(len(conf_te))])
    Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(Xb, yb, test_size=0.5, stratify=yb, random_state=seed)
    att = LogisticRegression().fit(Xa_tr, ya_tr)
    s = att.predict_proba(Xa_te)[:, 1]
    held = roc_auc_score(ya_te, s)
    boots = []
    for _ in range(1000):
        i = rng.integers(0, len(ya_te), len(ya_te))
        if len(np.unique(ya_te[i])) == 2: boots.append(roc_auc_score(ya_te[i], s[i]))
    return {"paper_protocol_auc": float(paper_auc), "heldout_balanced_auc": float(held),
            "heldout_auc_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n_attack_test": int(len(ya_te))}


def rdp_eps(sigma, q, steps, delta=DELTA):
    from dp_accounting import rdp, dp_event
    acc = rdp.RdpAccountant()
    acc.compose(dp_event.PoissonSampledDpEvent(q, dp_event.GaussianDpEvent(sigma)), steps)
    return float(acc.get_epsilon(delta))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "validation" / "outputs"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="1-epoch smoke run of every code path")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    t0 = time.time()
    ep_long = 1 if a.quick else 120
    ep_short = 1 if a.quick else 10
    seeds = [a.seed] if a.quick else [a.seed, a.seed + 1, a.seed + 2]
    eps_grid = [0.2, 2.0] if a.quick else [0.2, 0.5, 1.0, 2.0, 4.0, 8.0]

    D = A.load_dataset(a.seed)  # 12 predictors incl. loan_id, StandardScaler fit on train only
    X12, Xt12, y, yt = D["X_train"], D["X_test"], D["y_train"], D["y_test"]
    X11, Xt11 = np.ascontiguousarray(X12[:, 1:]), np.ascontiguousarray(Xt12[:, 1:])
    R = {"environment": {"python": platform.python_version(), "torch": torch.__version__,
                         "numpy": np.__version__, "pandas": pd.__version__},
         "dataset": {"sha256": A.sha256(A.DATA_PATH), "rows": int(len(D["frame"])),
                     "class_counts": D["class_counts"], "n_train": int(len(y)), "n_test": int(len(yt)),
                     "feature_names_12": D["feature_names"]},
         "settings": {"seed": a.seed, "clip_C": C, "delta": DELTA, "lr_paper": 0.001,
                      "short_epochs": ep_short, "long_epochs": ep_long, "quick": a.quick}}
    n = len(y)

    # E1 non-private baselines
    E1 = []
    for feats, (Xa, Xb) in [("12 (incl. loan_id)", (X12, Xt12)), ("11 (excl. loan_id)", (X11, Xt11))]:
        for b in [32, 64]:
            m, p_te = metrics(nonprivate(Xa, y, b, ep_long, a.seed), Xb, yt)
            E1.append({"features": feats, "batch": b, "epochs": ep_long, **m})
            print("E1", E1[-1], flush=True)
    pd.DataFrame(E1).to_csv(out / "E1_nonprivate.csv", index=False)

    # E2 Table 2 reconstruction (12-feature paper-faithful input)
    E2 = []
    for b in [1, 32, 64]:
        steps = math.ceil(n / b) * ep_short
        m, _ = metrics(train_dp(X12, y, b, ep_short, 0.001, a.seed + b, "laplace",
                                lambda B: ("laplace", C / 0.2)), Xt12, yt)
        E2.append({"variant": "(a) literal Algorithm 1: Laplace(C/eps)", "batch": b, "epochs": ep_short,
                   "per_update_eps_replace_one": 2 * 0.2 / b, "steps": steps, **m})
        m, _ = metrics(train_dp(X12, y, b, ep_short, 0.001, a.seed + b, "laplace",
                                lambda B: ("laplace", 2 * C / (B * 0.2))), Xt12, yt)
        E2.append({"variant": "(b) calibrated: Laplace(2C/(B*eps))", "batch": b, "epochs": ep_short,
                   "per_update_eps_replace_one": 0.2, "steps": steps, **m})
        m, _ = metrics(train_dp(X12, y, b, ep_short, 0.01, a.seed + b, "gaussian",
                                lambda B: ("gaussian", 1.1 * C / B), clip_norm="l2"), Xt12, yt)
        E2.append({"variant": "(c) DP-SGD comparator: L2 clip, sigma=1.1, lr=0.01", "batch": b, "epochs": ep_short,
                   "per_update_eps_replace_one": None, "steps": steps,
                   "rdp_total_eps_delta1e-5": rdp_eps(1.1, b / n, steps), **m})
        for r in E2[-3:]: print("E2", r, flush=True)
    pd.DataFrame(E2).to_csv(out / "E2_table2_reconstruction.csv", index=False)

    # E3 equal-privacy comparison (same clipping, sensitivity 2C/B, same per-update eps; Gaussian delta_step=delta/epochs)
    E3 = []
    for clip in ["l1", "l2"]:
        for b in [32, 64]:
            for e in eps_grid:
                sig = analytic_gaussian_sigma(e, DELTA / ep_short)
                for s in seeds:
                    for mech in ["laplace", "gaussian"]:
                        if mech == "laplace" and clip == "l2":
                            continue  # Laplace requires L1 sensitivity
                        fn = (lambda B, e=e: ("laplace", 2 * C / (B * e))) if mech == "laplace" else \
                             (lambda B, sig=sig: ("gaussian", sig * 2 * C / B))
                        m, _ = metrics(train_dp(X12, y, b, ep_short, 0.001, s * 1000 + b, mech, fn, clip_norm=clip), Xt12, yt)
                        E3.append({"clipping": clip.upper(), "mechanism": mech, "batch": b, "per_update_eps": e,
                                   "gaussian_sigma_multiplier": sig if mech == "gaussian" else None,
                                   "seed": s, "epochs": ep_short, "per_record_total_eps": e * ep_short, **m})
                print("E3", clip, b, e, flush=True)
    E3 = pd.DataFrame(E3); E3.to_csv(out / "E3_equal_privacy_raw.csv", index=False)
    summ = E3.groupby(["clipping", "mechanism", "batch", "per_update_eps"]).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
        roc_auc_mean=("roc_auc", "mean"), n_seeds=("seed", "count")).reset_index()
    summ.to_csv(out / "E3_equal_privacy_summary.csv", index=False)

    # E4 accounting
    E4 = []
    for b in [1, 32, 64]:
        for ep in [10, 120]:
            steps = math.ceil(n / b) * ep
            for e in [0.2, 0.5, 2.0, 8.0]:
                E4.append({"batch": b, "epochs": ep, "per_update_eps": e, "updates": steps,
                           "per_record_eps_disjoint_batches": e * ep,
                           "naive_all_updates_eps": e * steps})
    E4 = pd.DataFrame(E4); E4.to_csv(out / "E4_privacy_accounting.csv", index=False)
    dpsgd = [{"batch": b, "epochs": ep, "sigma": 1.1, "sampling_rate": b / n,
              "rdp_eps_delta1e-5": rdp_eps(1.1, b / n, math.ceil(n / b) * ep)} for b in [32, 64] for ep in [10]]
    pd.DataFrame(dpsgd).to_csv(out / "E4_dpsgd_rdp.csv", index=False)

    # E5 membership inference
    E5 = []
    for name, model in [
        ("non-private, B=32, %d epochs" % ep_long, nonprivate(X12, y, 32, ep_long, a.seed)),
        ("Laplace calibrated eps=0.2, B=32, %d epochs" % ep_short,
         train_dp(X12, y, 32, ep_short, 0.001, a.seed + 32, "laplace", lambda B: ("laplace", 2 * C / (B * 0.2)))),
        ("Laplace calibrated eps=2, B=32, %d epochs" % ep_short,
         train_dp(X12, y, 32, ep_short, 0.001, a.seed + 32, "laplace", lambda B: ("laplace", 2 * C / (B * 2.0)))),
    ]:
        m_te, p_te = metrics(model, Xt12, yt)
        p_tr = A.predict(model, X12, torch.device("cpu"))
        E5.append({"model": name, "test_accuracy": m_te["accuracy"], **mia(p_tr, p_te, y, yt, a.seed)})
        print("E5", E5[-1], flush=True)
    pd.DataFrame(E5).to_csv(out / "E5_membership_inference.csv", index=False)

    R.update({"E1": E1, "E2": E2, "E3_summary": summ.to_dict("records"), "E4_dpsgd": dpsgd,
              "E5": E5, "runtime_seconds": time.time() - t0})
    json.dump(R, open(out / "validation_results.json", "w"), indent=2, default=float)
    print("DONE", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
