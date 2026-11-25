"""Command line: python -m claimfraud --data data/carclaims.csv  (then: mlflow ui --backend-store-uri sqlite:///mlflow.db)"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402

from . import data, models  # noqa: E402


def gpu_available() -> bool:
    try:
        from xgboost import XGBClassifier

        XGBClassifier(device="cuda", n_estimators=1).fit(np.zeros((4, 1)), [0, 1, 0, 1])
        return True
    except Exception:
        return False


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description="Insurance claim fraud detection")
    p.add_argument("--data", default="data/carclaims.csv")
    p.add_argument("--out", default="results")
    p.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--only", nargs="*")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("insurance-claim-fraud")
    device = "cuda" if (args.device == "auto" and gpu_available()) or args.device == "cuda" else "cpu"

    feats = data.build_features(data.load(args.data))
    train, valid, test = data.time_split(feats)
    cols = data.NUMERIC + data.CATEGORICAL
    full_train = pd.concat([train, valid])
    pos_weight = float((full_train[data.TARGET] == 0).sum() / (full_train[data.TARGET] == 1).sum())
    print(f"train {len(train):,} (1994)  validation {len(valid):,} (1995)  test {len(test):,} (1996)  "
          f"fraud {feats[data.TARGET].mean():.1%}  device {device}")

    rows, curves, fitted = [], {}, {}
    for name, pipe in models.candidates(pos_weight, device).items():
        if args.only and name not in args.only:
            continue
        t0 = time.time()
        # 1) fit on 1994, choose the decision threshold on 1995;  2) refit on 1994-1995, evaluate on 1996.
        early = clone(pipe).fit(train[cols], train[data.TARGET])
        threshold = models.best_f1_threshold(valid[data.TARGET], early.predict_proba(valid[cols])[:, 1])
        final = clone(pipe).fit(full_train[cols], full_train[data.TARGET])
        scores = final.predict_proba(test[cols])[:, 1]
        m = models.metrics(test[data.TARGET], scores, threshold)
        from sklearn.metrics import average_precision_score, roc_auc_score

        m["pr_auc_95ci"] = list(models.bootstrap_ci(test[data.TARGET], scores, average_precision_score))
        m["roc_auc_95ci"] = list(models.bootstrap_ci(test[data.TARGET], scores, roc_auc_score))
        m["fit_seconds"] = round(time.time() - t0, 1)
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": name, "train_years": "1994-1995", "test_year": 1996,
                               "device": device if "XGBoost" in name else "cpu"})
            mlflow.log_metrics({k: v for k, v in m.items() if not isinstance(v, list)})
        rows.append({"model": name, **m})
        curves[name] = scores
        fitted[name] = final
        print(f"{name:40s} PR-AUC {m['pr_auc']:.3f}  ROC-AUC {m['roc_auc']:.3f}  F1 {m['f1']:.3f}  "
              f"recall {m['recall']:.2f}  fraud caught reviewing top 20%: {m['fraud_caught_reviewing_20pct']:.0%}")

    table = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    plot_capture(test[data.TARGET].to_numpy(), curves, out / "fraud_caught_by_review_share.png")
    shap_top = explain(fitted["XGBoost"], test[cols], out / "shap_importance.png") if "XGBoost" in fitted else None
    summary = {
        "rows": {"train_1994": len(train), "validation_1995": len(valid), "test_1996": len(test)},
        "fraud_rate": {"all": round(float(feats[data.TARGET].mean()), 4),
                       "test": round(float(test[data.TARGET].mean()), 4)},
        "device": device, "results": table.to_dict(orient="records"), "best_model": table.iloc[0]["model"],
        "shap_top_features": shap_top,
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    return summary


def plot_capture(y, curves, path) -> None:
    """How much fraud investigators find as they review more of the highest-scored claims."""
    shares = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for name, scores in curves.items():
        caught = [models.fraud_caught_when_reviewing(y, scores, s) if s > 0 else 0 for s in shares]
        ax.plot(shares * 100, np.array(caught) * 100, label=name, linewidth=2.2 if "XGBoost" in name else 1.3)
    ax.plot([0, 100], [0, 100], color="grey", linestyle=":", label="Random order")
    ax.axvline(20, color="#999", linewidth=0.8)
    ax.set_xlabel("Claims reviewed, highest risk first (%)")
    ax.set_ylabel("Fraud found (%)")
    ax.set_title("Test year 1996: fraud found per share of claims reviewed")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def explain(pipe, X, path) -> list:
    import shap

    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    Z = prep.transform(X)
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    model.set_params(device="cpu")
    values = shap.TreeExplainer(model).shap_values(Z)
    imp = pd.Series(np.abs(values).mean(axis=0), index=names).sort_values(ascending=False)
    top = imp.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(top.index, top.values, color="#2f6fb0")
    ax.set_xlabel("Mean |SHAP value| (impact on the fraud score)")
    ax.set_title("What drives the XGBoost claim-fraud score")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [{"feature": k, "mean_abs_shap": round(float(v), 4)} for k, v in imp.head(10).items()]


if __name__ == "__main__":
    main()
