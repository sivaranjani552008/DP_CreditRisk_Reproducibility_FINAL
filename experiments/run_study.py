#!/usr/bin/env python3
"""
run_study.py - executes the pre-specified Laplace-vs-Gaussian settings study
(experiments/study_config.json) with src/dp_ppnn.py and writes every run.

    python experiments/run_study.py --out results/study            # full (~35 min, 2 cores)
    python experiments/run_study.py --out /tmp/q --quick            # smoke test (<1 min)

Outputs: runs.csv (one row per training run / noise draw set), noise_table.csv (S1),
ledger fields for every run, environment.json.  Summaries: experiments/summarize.py.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, itertools, json, math, platform, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import dp_ppnn as P  # noqa: E402

CFG = json.load(open(ROOT / "experiments" / "study_config.json"))
_DATA = {}


def data():
    if not _DATA:
        df = P.load_frame()
        Xtr, Xte, ytr, yte, _, _ = P.split(df, CFG["split_seed"])
        Xs, Xts = P.standardize(Xtr, Xte)
        _DATA.update(Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, Xs=Xs, Xts=Xts)
    return _DATA


ARM = {"laplace_l1": ("laplace", "l1"), "gaussian_l1": ("gaussian", "l1"), "gaussian_l2": ("gaussian", "l2")}


def run_gradient(t):
    D = data()
    kw = dict(clip_norm=CFG["clip_norm_C"], delta=CFG["delta"], n_params=2945, shards=t.get("shards", 1))
    if t["study"] == "S2":
        mech, clip = t["mech"], "l2"
        plan = P.calibrate(mech, "literal", t["eps"], t["batch"], t["epochs"], clip=clip,
                           gaussian_multiplier=t.get("sigma"), **kw)
    else:
        mech, clip = ARM[t["arm"]]
        plan = P.calibrate(mech, t["mode"], t["eps"], t["batch"], t["epochs"], clip=clip, **kw)
    t0 = time.time()
    model = P.train(D["Xs"], D["ytr"], batch=t["batch"], epochs=t["epochs"], lr=t["lr"], seed=t["seed"],
                    plan=plan, clip=clip, C=CFG["clip_norm_C"], arch=t.get("arch", "report"),
                    shards=t.get("shards", 1))
    m = P.metrics(D["yte"], model.predict_proba(D["Xts"]))
    return {**t, "mechanism": mech, "clip": clip, "noise_scale": plan.scale,
            **{f"ledger_{k}": v for k, v in plan.ledger.items()}, **m, "seconds": time.time() - t0}


def run_nonprivate(t):
    D = data()
    model = P.train(D["Xs"], D["ytr"], batch=t["batch"], epochs=t["epochs"], lr=t["lr"], seed=t["seed"],
                    plan=P.NoisePlan("none", 0.0), clip=None, arch=t["arch"])
    out = {**t, **P.metrics(D["yte"], model.predict_proba(D["Xts"]))}
    if t["arch"] == "report" and t["batch"] == 32:  # base models for S7 (output perturbation)
        out["_test_prob"] = model.predict_proba(D["Xts"]).tolist()
    return out


def run_input(t):
    D = data()
    rng = np.random.default_rng([t["seed"], 7])
    Xtr01, Xte01 = P.to_unit_box(D["Xtr"]), P.to_unit_box(D["Xte"])
    Ntr, led = P.input_perturbation(Xtr01, t["mech"], t["eps"], rng, t["convention"], CFG["delta"])
    Nte, _ = P.input_perturbation(Xte01, t["mech"], t["eps"], rng, t["convention"], CFG["delta"])
    Ztr, Zte = P.standardize(Ntr, Nte)          # post-processing of the released noisy data
    _, Zclean = P.standardize(Ntr, Xte01)
    c = CFG["S8_input_perturbation"]["train"]
    model = P.train(Ztr, D["ytr"], batch=c["batch"], epochs=c["epochs"], lr=c["lr"], seed=t["seed"],
                    plan=P.NoisePlan("none", 0.0), clip=None)
    rows = []
    for which, Z in [("noisy", Zte), ("clean", Zclean)]:
        rows.append({**t, "test_inputs": which, **{f"ledger_{k}": v for k, v in led.items()},
                     **P.metrics(D["yte"], model.predict_proba(Z))})
    return rows


def dispatch(t):
    if t["study"] == "S0":
        return [run_nonprivate(t)]
    if t["study"] == "S8":
        return run_input(t)
    return [run_gradient(t)]


def tasks(quick=False):
    S = CFG["seeds"][:1] if quick else CFG["seeds"]
    T = []
    c = CFG["S0_nonprivate"]
    for arch, b, s in itertools.product(c["arch"], c["batch"], S):
        T.append(dict(study="S0", arch=arch, batch=b, epochs=2 if quick else c["epochs"], lr=c["lr"], seed=s))
    c = CFG["S2_table2_literal"]
    for b, s in itertools.product(c["batch"], S):
        if quick and b == 1: continue
        ep = 1 if quick else c["epochs"]
        T.append(dict(study="S2", mech="laplace", batch=b, epochs=ep, lr=c["laplace"]["lr"], eps=c["eps"], seed=s))
        T.append(dict(study="S2", mech="gaussian", batch=b, epochs=ep, lr=c["dpsgd"]["lr"], eps=c["eps"],
                      sigma=c["dpsgd"]["sigma"], seed=s))
    c = CFG["S3_gradient_per_update"]
    for b, e, lr, arm, s in itertools.product(c["batch"], c["eps"], c["lr"], c["arms"], S):
        if quick and (b == 1 or e not in (2.0,)): continue
        T.append(dict(study="S3", mode="per_update", arm=arm, batch=b, eps=e, lr=lr,
                      epochs=1 if quick else c["epochs"], seed=s))
    c = CFG["S4_gradient_total_budget"]
    for b, e, ep, lr, arm, s in itertools.product(c["batch"], c["eps_total"], c["epochs"], c["lr"], c["arms"], S):
        if quick and (ep > 1 or e != 8.0): continue
        T.append(dict(study="S4", mode="total", arm=arm, batch=b, eps=e, lr=lr, epochs=ep, seed=s))
    c = CFG["S5_algorithm2_parallel"]
    for np_, b, e, lr, arm, s in itertools.product(c["shards"], c["batch"], c["eps"], c["lr"], c["arms"], S):
        if quick and e != 8.0: continue
        T.append(dict(study="S5", mode="per_update", arm=arm, shards=np_, batch=b, eps=e, lr=lr,
                      epochs=1 if quick else c["epochs"], seed=s))
    c = CFG["S6_utility_equivalent_eps"]
    for b, e, lr, arm, s in itertools.product(c["batch"], c["eps"], c["lr"], c["arms"], c["seeds"]):
        if quick: continue
        T.append(dict(study="S6", mode="per_update", arm=arm, batch=b, eps=e, lr=lr, epochs=c["epochs"], seed=s))
    c = CFG["S8_input_perturbation"]
    for conv, grid in c["conventions"].items():
        grid = CFG["article_eps_grid"] if grid == "article_eps_grid" else grid
        for e, mech, s in itertools.product(grid, ["laplace", "gaussian"], S):
            if quick and e not in (1.0, 12.0): continue
            T.append(dict(study="S8", convention=conv, eps=e, mech=mech, seed=s))
    # longest first for load balance
    T.sort(key=lambda t: -(t.get("epochs", 120) * 3415 / t.get("batch", 32)))
    return T


def noise_table():
    rows = []
    for e in CFG["article_eps_grid"]:
        for d in [1e-5, 1e-6]:
            s = P.analytic_gaussian_sigma(e, d)
            rows.append({"table": "single_release", "eps": e, "delta": d, "laplace_std": math.sqrt(2) / e,
                         "gaussian_std": s, "gaussian_over_laplace": s * e / math.sqrt(2)})
    for K in [1, 2, 5, 10, 30, 120]:
        for e in [2.0, 4.0, 8.0, 16.0]:
            lap = math.sqrt(2) * K / e
            gau = math.sqrt(K) * P.analytic_gaussian_sigma(e, 1e-5)
            rows.append({"table": "total_budget", "epochs_K": K, "eps": e, "delta": 1e-5, "laplace_std": lap,
                         "gaussian_std": gau, "gaussian_over_laplace": gau / lap})
    return pd.DataFrame(rows)


def output_study(s0_rows, quick=False):
    D = data()
    grid = CFG["article_eps_grid"]
    draws = 5 if quick else CFG["S7_output_perturbation"]["noise_draws"]
    rows = []
    for r in s0_rows:
        if "_test_prob" not in r: continue
        prob = np.asarray(r["_test_prob"])
        for e in grid:
            for mech in ["laplace", "gaussian"]:
                rng = np.random.default_rng([r["seed"], int(e * 100), 11])   # same stream for both arms
                acc, auc, bal = [], [], []
                for _ in range(draws):
                    noisy, scale = P.output_perturbation(prob, mech, e, rng, CFG["delta"])
                    m = P.metrics(D["yte"], noisy)
                    acc.append(m["accuracy"]); auc.append(m["roc_auc"]); bal.append(m["balanced_accuracy"])
                rows.append({"study": "S7", "seed": r["seed"], "eps": e, "mechanism": mech, "noise_scale": scale,
                             "noise_std": scale * math.sqrt(2) if mech == "laplace" else scale,
                             "accuracy": float(np.mean(acc)), "balanced_accuracy": float(np.mean(bal)),
                             "roc_auc": float(np.mean(auc)), "draws": draws,
                             "base_accuracy": r["accuracy"], "base_roc_auc": r["roc_auc"]})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "study"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--procs", type=int, default=max(1, os.cpu_count() or 1))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    T = tasks(a.quick)
    print(f"{len(T)} tasks", flush=True)
    rows = []
    with Pool(a.procs) as pool:
        for i, res in enumerate(pool.imap_unordered(dispatch, T, chunksize=1)):
            rows.extend(res)
            if i % 50 == 0:
                print(f"{i + 1}/{len(T)}  {time.time() - t0:.0f}s", flush=True)
    s0 = [r for r in rows if r["study"] == "S0"]
    out7 = output_study(s0, a.quick)
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    df = pd.concat([df, out7], ignore_index=True)
    df.to_csv(out / "runs.csv", index=False)
    noise_table().to_csv(out / "noise_table.csv", index=False)
    import scipy, sklearn
    json.dump({"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
               "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
               "dataset_sha256": P.sha256(P.DATA_PATH), "tasks": len(T), "quick": a.quick,
               "runtime_seconds": time.time() - t0, "config": CFG},
              open(out / "environment.json", "w"), indent=2)
    print(f"DONE {len(df)} rows in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
