import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from claimfraud import data, models
from claimfraud.__main__ import main


def claims(n=3000, seed=0):
    """Synthetic claims with the real schema; fraud is likelier when the policy holder is at fault on an
    All Perils policy."""
    rng = np.random.default_rng(seed)
    pick = lambda d: rng.choice(list(d), n)  # noqa: E731
    fault = rng.choice(["Policy Holder", "Third Party"], n, p=[0.7, 0.3])
    base = rng.choice(["Liability", "Collision", "All Perils"], n)
    p = 0.01 + 0.15 * (fault == "Policy Holder") * (base != "Liability") + 0.25 * (fault == "Policy Holder") * (base == "All Perils")
    months = list(data.MONTHS)
    df = pd.DataFrame({
        "Month": rng.choice(months, n), "WeekOfMonth": rng.integers(1, 6, n), "DayOfWeek": "Monday",
        "Make": rng.choice(["Honda", "Toyota", "Ford"], n), "AccidentArea": rng.choice(["Urban", "Rural"], n),
        "DayOfWeekClaimed": "Tuesday", "MonthClaimed": rng.choice(months, n), "WeekOfMonthClaimed": rng.integers(1, 6, n),
        "Sex": rng.choice(["Male", "Female"], n), "MaritalStatus": "Married", "Age": rng.integers(0, 80, n),
        "Fault": fault, "PolicyType": "Sedan - Collision", "VehicleCategory": rng.choice(["Sedan", "Sport"], n),
        "PolicyNumber": np.arange(1, n + 1), "RepNumber": rng.integers(1, 17, n), "Deductible": 400,
        "DriverRating": rng.integers(1, 5, n), "PoliceReportFiled": "No", "WitnessPresent": "No",
        "AgentType": "External", "Year": np.array([1994, 1995, 1996])[np.arange(n) * 3 // n], "BasePolicy": base,
        "FraudFound": np.where(rng.random(n) < p, "Yes", "No"),
    })
    for c, mapping in data.ORDINAL.items():
        df[c] = pick(mapping)
    return df


def test_range_text_becomes_ordered_numbers():
    df = claims(30)
    df.loc[0, ["VehiclePrice", "AgeOfVehicle", "AddressChange-Claim"]] = ["more than 69,000", "new", "under 6 months"]
    df.loc[0, "Age"] = 0
    df.loc[0, ["Month", "MonthClaimed"]] = ["Dec", "Jan"]
    f = data.build_features(df)
    assert f.loc[0, "VehiclePrice"] == 6 and f.loc[0, "AgeOfVehicle"] == 0 and f.loc[0, "AddressChange-Claim"] == 4
    assert np.isnan(f.loc[0, "Age"])          # 0 is a placeholder for ages 16-17, not a real age
    assert f.loc[0, "claim_lag_months"] == 1  # December accident claimed in January
    assert set(f["FraudFound"].unique()) <= {0, 1}


def test_unexpected_range_value_is_rejected():
    df = claims(10)
    df.loc[0, "PastNumberOfClaims"] = "dozens"
    with pytest.raises(ValueError, match="PastNumberOfClaims"):
        data.build_features(df)


def test_missing_column_is_rejected(tmp_path):
    path = tmp_path / "c.csv"
    claims(10).drop(columns=["Fault"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Fault"):
        data.load(path)


def test_time_split_by_year():
    f = data.build_features(claims())
    train, valid, test = data.time_split(f)
    assert set(train["Year"]) == {1994} and set(valid["Year"]) == {1995} and set(test["Year"]) == {1996}
    assert len(train) + len(valid) + len(test) == len(f)


def test_review_share_metric():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 1])
    s = np.array([.9, .1, .2, .8, .3, .1, .2, .1, .4, .05])
    assert models.fraud_caught_when_reviewing(y, s, 0.2) == pytest.approx(2 / 3)
    assert models.fraud_caught_when_reviewing(y, s, 1.0) == 1


def test_bootstrap_interval_contains_the_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = y * 0.5 + rng.random(500)
    lo, hi = models.bootstrap_ci(y, s, roc_auc_score, n=300)
    assert lo < roc_auc_score(y, s) < hi


def test_end_to_end_finds_planted_signal(tmp_path):
    path = tmp_path / "claims.csv"
    claims().to_csv(path, index=False)
    out = tmp_path / "results"
    summary = main(["--data", str(path), "--out", str(out), "--device", "cpu",
                    "--tracking-uri", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"])
    assert len(summary["results"]) == 4
    assert all(r["roc_auc"] > 0.65 for r in summary["results"])
    assert json.loads((out / "results.json").read_text())["best_model"]
    assert (out / "fraud_caught_by_review_share.png").exists() and (out / "shap_importance.png").exists()
    top = [t["feature"] for t in summary["shap_top_features"][:4]]
    assert "is_Fault" in top
