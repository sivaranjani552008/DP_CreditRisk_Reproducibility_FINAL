#!/usr/bin/env python3
"""Focused, pre-specified Laplace-vs-Gaussian claim-support experiment.

This script does NOT encode the desired/published accuracies. It executes
the same clipped-gradient reconstruction for the four configurations
specified in the audit request and reports the observed metrics.
"""
from pathlib import Path
import argparse, json, time, numpy as np, pandas as pd, torch, random
from reproduce_article_claim_audit import (
    load_dataset, train_private, predict, metric_bundle, set_seed
)

ROOT=Path(__file__).resolve().parents[1]

CONFIGS=[(32,8.0),(32,32.0),(64,8.0),(64,32.0)]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--epochs",type=int,default=10)
    ap.add_argument("--learning-rate",type=float,default=0.001)
    ap.add_argument("--device",default="cpu",choices=["cpu","cuda"])
    ap.add_argument("--out",default=str(ROOT/"results"/"claim_support_audit"))
    args=ap.parse_args()
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device=torch.device(args.device)
    data=load_dataset(args.seed)
    rows=[]
    for batch,eps in CONFIGS:
        for mech in ["laplace","gaussian"]:
            seed=args.seed + batch + int(eps*1000) + (10000 if mech=="gaussian" else 0)
            model,hist=train_private(
                data["X_train"], data["y_train"],
                batch_size=batch, epochs=args.epochs,
                learning_rate=args.learning_rate, seed=seed,
                device=device, mechanism=mech, epsilon=eps,
                clipping_norm=1.0, gaussian_sigma=1.1, norm_type="l1"
            )
            p=predict(model,data["X_test"],device)
            m=metric_bundle(np.asarray(data["y_test"],dtype=np.int64),p)
            rows.append({
                "configuration":f"B={batch}, epsilon={eps:g}",
                "mechanism":mech,"batch_size":batch,"epsilon":eps,
                "epochs":args.epochs,"seed":seed,
                "accuracy_percent":100*m["accuracy"],
                "roc_auc":m["roc_auc"],"loss":m["log_loss"]
            })
    raw=pd.DataFrame(rows)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    raw.to_csv(out/"claim_support_raw_results.csv",index=False)
    comp=raw.pivot_table(index=["configuration","batch_size","epsilon","epochs"],
                         columns="mechanism",values="accuracy_percent",aggfunc="first").reset_index()
    comp["laplace_minus_gaussian_pp"]=comp["laplace"]-comp["gaussian"]
    comp["laplace_higher"]=comp["laplace_minus_gaussian_pp"]>0
    comp.to_csv(out/"claim_support_comparison.csv",index=False)
    print(comp.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\nNo published/target accuracy was used in computation.")
if __name__=="__main__":
    main()
