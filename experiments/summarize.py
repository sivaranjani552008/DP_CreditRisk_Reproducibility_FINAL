#!/usr/bin/env python3
"""
summarize.py - paired statistics, tables and figures for the settings study.

    python experiments/summarize.py --runs results/study

Paired design: for a fixed configuration and seed, the Laplace and Gaussian runs share the
initialisation and minibatch order (and, for S7/S8, the random stream), so differences are
computed per seed and summarised with a two-sided 95% t-interval over the 5 seeds.
Verdict per configuration: "Laplace" if the interval lies above 0, "Gaussian" if below 0,
"tie" otherwise.  Primary metric ROC-AUC; accuracy and balanced accuracy are also reported.
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

MET = ["roc_auc", "accuracy", "balanced_accuracy"]


def paired(df, key, a_filter, b_filter, label):
    a = df[a_filter].set_index(key + ["seed"])[MET]
    b = df[b_filter].set_index(key + ["seed"])[MET]
    d = (a - b).dropna()
    rows = []
    for k, g in d.groupby(level=list(range(len(key)))):
        k = k if isinstance(k, tuple) else (k,)
        r = dict(zip(key, k)); r["comparison"] = label; r["n"] = len(g)
        for m in MET:
            x = g[m].to_numpy()
            mu, sd = x.mean(), x.std(ddof=1) if len(x) > 1 else 0.0
            h = stats.t.ppf(0.975, len(x) - 1) * sd / np.sqrt(len(x)) if len(x) > 1 else np.nan
            r[f"diff_{m}"] = mu; r[f"ci_lo_{m}"] = mu - h; r[f"ci_hi_{m}"] = mu + h
            r[f"verdict_{m}"] = "Laplace" if mu - h > 0 else ("Gaussian" if mu + h < 0 else "tie")
        for side, f in [("A", a_filter), ("B", b_filter)]:
            sub = df[f]
            for kk, v in zip(key, k):
                sub = sub[sub[kk] == v]
            for m in MET:
                r[f"{side}_{m}"] = sub[m].mean()
        rows.append(r)
    return pd.DataFrame(rows)


def counts(t, m="roc_auc"):
    v = t[f"verdict_{m}"].value_counts()
    return {"configs": int(len(t)), "laplace": int(v.get("Laplace", 0)), "gaussian": int(v.get("Gaussian", 0)),
            "tie": int(v.get("tie", 0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/study")
    a = ap.parse_args()
    R = Path(a.runs)
    df = pd.read_csv(R / "runs.csv")
    out = {}
    T = {}

    s0 = df[df.study == "S0"]
    T["S0"] = s0.groupby(["arch", "batch"])[MET + ["log_loss"]].agg(["mean", "std"]).reset_index()
    T["S0"].columns = ["_".join(c).strip("_") for c in T["S0"].columns]

    s2 = df[df.study == "S2"]
    T["S2"] = s2.groupby(["mechanism", "batch"]).agg(
        acc_mean=("accuracy", "mean"), acc_sd=("accuracy", "std"), auc_mean=("roc_auc", "mean"),
        auc_sd=("roc_auc", "std"), bal_mean=("balanced_accuracy", "mean"), loss_mean=("log_loss", "mean"),
        eps_actual_per_update=("ledger_per_update_eps_actual", "mean"),
        eps_actual_total=("ledger_per_record_total_eps", "mean"), n=("seed", "count")).reset_index()
    # DP-SGD arm: Renyi-DP accountant with Poisson subsampling q = B/n (the moments-accountant
    # convention of TensorFlow Privacy; training itself uses shuffled disjoint minibatches).
    try:
        from dp_accounting import rdp, dp_event
        n_train = 3415

        def rdp_eps(b, ep):
            acc = rdp.RdpAccountant()
            acc.compose(dp_event.PoissonSampledDpEvent(b / n_train, dp_event.GaussianDpEvent(1.1)),
                        int(np.ceil(n_train / b)) * ep)
            return float(acc.get_epsilon(1e-5))
        T["S2"]["rdp_total_eps_subsampled"] = [rdp_eps(int(b), 10) if m == "gaussian" else np.nan
                                               for m, b in zip(T["S2"].mechanism, T["S2"].batch)]
    except ImportError:
        pass

    s3 = df[df.study == "S3"]
    key3 = ["batch", "eps", "lr"]
    lap3 = s3.arm == "laplace_l1"
    T["S3_matched"] = paired(s3, key3, lap3, s3.arm == "gaussian_l1", "Laplace-L1 vs Gaussian-L1 (matched)")
    T["S3_natural"] = paired(s3, key3, lap3, s3.arm == "gaussian_l2", "Laplace-L1 vs Gaussian-L2 (natural)")
    s4 = df[df.study == "S4"]
    key4 = ["batch", "eps", "epochs", "lr"]
    T["S4_matched"] = paired(s4, key4, s4.arm == "laplace_l1", s4.arm == "gaussian_l1", "total, matched")
    T["S4_natural"] = paired(s4, key4, s4.arm == "laplace_l1", s4.arm == "gaussian_l2", "total, natural")
    s5 = df[df.study == "S5"]
    T["S5"] = paired(s5, ["shards", "batch", "eps", "lr"], s5.arm == "laplace_l1", s5.arm == "gaussian_l1",
                     "Algorithm 2, matched")
    s6 = df[df.study == "S6"]
    T["S6"] = s6.groupby(["batch", "lr", "eps"])[MET].mean().reset_index()
    s7 = df[df.study == "S7"]
    T["S7"] = paired(s7, ["eps"], s7.mechanism == "laplace", s7.mechanism == "gaussian", "output")
    s8 = df[df.study == "S8"]
    T["S8"] = paired(s8, ["convention", "test_inputs", "eps"], s8.mech == "laplace", s8.mech == "gaussian", "input")

    for k, t in T.items():
        t.to_csv(R / f"summary_{k}.csv", index=False)

    out["S3_matched"] = counts(T["S3_matched"]); out["S3_natural"] = counts(T["S3_natural"])
    out["S3_matched_bal_acc"] = counts(T["S3_matched"], "balanced_accuracy")
    out["S4_matched"] = counts(T["S4_matched"]); out["S4_natural"] = counts(T["S4_natural"])
    out["S5"] = counts(T["S5"]); out["S7"] = counts(T["S7"]); out["S7_acc"] = counts(T["S7"], "accuracy")
    for conv in ["per_feature", "record"]:
        for ti in ["noisy", "clean"]:
            t = T["S8"][(T["S8"].convention == conv) & (T["S8"].test_inputs == ti)]
            out[f"S8_{conv}_{ti}"] = counts(t)
    for K in sorted(T["S4_matched"].epochs.unique()):
        out[f"S4_matched_K{int(K)}"] = counts(T["S4_matched"][T["S4_matched"].epochs == K])
    # privacy-utility trade-off: does mean ROC-AUC rise with eps on every curve?
    g = s3.groupby(["arm", "batch", "lr", "eps"])["roc_auc"].mean().reset_index()
    mono = [bool(np.all(np.diff(d.sort_values("eps").roc_auc.values) > 0)) for _, d in g.groupby(["arm", "batch", "lr"])]
    out["tradeoff_monotone_auc_curves"] = {"curves": len(mono), "monotone": int(sum(mono))}
    o7 = T["S7"].sort_values("eps")
    out["tradeoff_output_monotone"] = bool(np.all(np.diff(o7.A_accuracy.values) > 0))
    # loss: Laplace vs matched Gaussian (Table 2 / Fig. 13 compare loss)
    lp = s3[s3.arm == "laplace_l1"].set_index(key3 + ["seed"])["log_loss"]
    gp = s3[s3.arm == "gaussian_l1"].set_index(key3 + ["seed"])["log_loss"]
    dd = (lp - gp).dropna()
    lo = hi = low_mean = 0
    for _, x in dd.groupby(level=[0, 1, 2]):
        h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
        low_mean += x.mean() < 0; lo += x.mean() + h < 0; hi += x.mean() - h > 0
    out["S3_matched_loss"] = {"configs": int(dd.groupby(level=[0, 1, 2]).ngroups), "laplace_lower_mean": int(low_mean),
                              "laplace_sig_lower": int(lo), "laplace_sig_higher": int(hi)}
    json.dump(out, open(R / "summary.json", "w"), indent=2)
    print(json.dumps(out, indent=1))
    make_figures(T, R)


def make_figures(T, R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False})

    # Figure 1: gradient stage, per-update eps, AUC vs eps (lr 0.01), B = 1/32/64
    m, n = T["S3_matched"], T["S3_natural"]
    fig, axs = plt.subplots(1, 3, figsize=(9.2, 3.0), sharey=True)
    for ax, b in zip(axs, [1, 32, 64]):
        mm = m[(m.batch == b) & (m.lr == 0.01)].sort_values("eps")
        nn = n[(n.batch == b) & (n.lr == 0.01)].sort_values("eps")
        ax.plot(mm.eps, mm.A_roc_auc, "-o", color=BLUE, lw=2, ms=5, label="Laplace, L1 clip")
        ax.plot(mm.eps, mm.B_roc_auc, "-s", color=ORANGE, lw=2, ms=5, label="Gaussian, L1 clip (matched)")
        ax.plot(nn.eps, nn.B_roc_auc, "--^", color=AQUA, lw=2, ms=5, label="Gaussian, L2 clip (DP-SGD)")
        ax.set_xscale("log", base=2); ax.set_xticks([0.2, 0.5, 1, 2, 4, 8]); ax.set_xticklabels(["0.2", "0.5", "1", "2", "4", "8"])
        ax.set_title(f"B = {b}", color=INK); ax.set_xlabel("per-update ε"); ax.grid(axis="y", color="#e6e5e0", lw=0.6)
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
    axs[0].set_ylabel("test ROC-AUC (mean of 5 seeds)")
    axs[0].legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(R / "fig_gradient_per_update.png", dpi=200); plt.close(fig)

    # Figure 2: output (Algorithm 3) and input (per-feature, noisy test) stages
    fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.0))
    o = T["S7"].sort_values("eps")
    axs[0].plot(o.eps, o.A_accuracy, "-o", color=BLUE, lw=2, ms=5, label="Laplace")
    axs[0].plot(o.eps, o.B_accuracy, "-s", color=ORANGE, lw=2, ms=5, label="Gaussian (δ = 1e-5)")
    axs[0].set_title("Output privacy (Algorithm 3)", color=INK); axs[0].set_ylabel("accuracy of private predictions")
    i = T["S8"]; i = i[(i.convention == "per_feature") & (i.test_inputs == "noisy")].sort_values("eps")
    axs[1].plot(i.eps, i.A_roc_auc, "-o", color=BLUE, lw=2, ms=5, label="Laplace")
    axs[1].plot(i.eps, i.B_roc_auc, "-s", color=ORANGE, lw=2, ms=5, label="Gaussian (δ = 1e-5)")
    axs[1].set_title("Input privacy, per-feature ε (Eq. 2)", color=INK); axs[1].set_ylabel("test ROC-AUC")
    for ax in axs:
        ax.set_xscale("log", base=2); ax.set_xticks([0.2, 0.5, 1, 2, 4, 8]); ax.set_xticklabels(["0.2", "0.5", "1", "2", "4", "8"])
        ax.set_xlabel("ε"); ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    axs[0].legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(R / "fig_output_input.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
