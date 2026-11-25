"""Loading the vehicle insurance claims data and turning its text ranges into features."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "FraudFound"
MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov",
                                       "Dec"], start=1)}

# Range columns stored as text: mapped to ordered numbers so models can use their order.
ORDINAL = {
    "VehiclePrice": {"less than 20,000": 1, "20,000 to 29,000": 2, "30,000 to 39,000": 3, "40,000 to 59,000": 4,
                     "60,000 to 69,000": 5, "more than 69,000": 6},
    "Days:Policy-Accident": {"none": 0, "1 to 7": 1, "8 to 15": 2, "15 to 30": 3, "more than 30": 4},
    "Days:Policy-Claim": {"none": 0, "8 to 15": 2, "15 to 30": 3, "more than 30": 4},
    "PastNumberOfClaims": {"none": 0, "1": 1, "2 to 4": 2, "more than 4": 3},
    "AgeOfVehicle": {"new": 0, "2 years": 2, "3 years": 3, "4 years": 4, "5 years": 5, "6 years": 6, "7 years": 7,
                     "more than 7": 8},
    "AgeOfPolicyHolder": {"16 to 17": 1, "18 to 20": 2, "21 to 25": 3, "26 to 30": 4, "31 to 35": 5, "36 to 40": 6,
                          "41 to 50": 7, "51 to 65": 8, "over 65": 9},
    "NumberOfSuppliments": {"none": 0, "1 to 2": 1, "3 to 5": 2, "more than 5": 3},
    # How recently the address changed before the claim: larger means more recent.
    "AddressChange-Claim": {"no change": 0, "4 to 8 years": 1, "2 to 3 years": 2, "1 year": 3, "under 6 months": 4},
    "NumberOfCars": {"1 vehicle": 1, "2 vehicles": 2, "3 to 4": 3, "5 to 8": 4, "more than 8": 5},
}
BINARY = {
    "Sex": "Male", "Fault": "Policy Holder", "PoliceReportFiled": "Yes", "WitnessPresent": "Yes",
    "AgentType": "Internal", "AccidentArea": "Urban",
}
CATEGORICAL = ["Make", "MaritalStatus", "VehicleCategory", "BasePolicy", "DayOfWeek", "DayOfWeekClaimed"]
NUMERIC = (["Age", "Deductible", "DriverRating", "WeekOfMonth", "WeekOfMonthClaimed", "month", "month_claimed",
            "claim_lag_months"] + list(ORDINAL) + [f"is_{c}" for c in BINARY])
# Dropped: PolicyNumber and RepNumber are identifiers; PolicyType repeats VehicleCategory + BasePolicy;
# Year is used only to split the data by time.


def load(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = set(ORDINAL) | set(BINARY) | set(CATEGORICAL) | {"Age", "Month", "MonthClaimed", "Year", TARGET,
                                                                 "PolicyNumber"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"claims file is missing columns: {missing}")
    return df.sort_values("PolicyNumber").reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    # Age 0 is recorded for policy holders aged 16-17: a placeholder, not an age.
    out["Age"] = df["Age"].replace(0, np.nan).astype(float)
    for c in ("Deductible", "DriverRating", "WeekOfMonth", "WeekOfMonthClaimed"):
        out[c] = df[c].astype(float)
    out["month"] = df["Month"].map(MONTHS)
    out["month_claimed"] = df["MonthClaimed"].map(MONTHS)  # one record has "0": becomes missing
    out["claim_lag_months"] = (out["month_claimed"] - out["month"]) % 12
    for c, mapping in ORDINAL.items():
        unknown = set(df[c].dropna().unique()) - set(mapping)
        if unknown:
            raise ValueError(f"unexpected values in {c}: {sorted(unknown)}")
        out[c] = df[c].map(mapping).astype(float)
    for c, positive in BINARY.items():
        out[f"is_{c}"] = (df[c] == positive).astype(float)
    for c in CATEGORICAL:
        out[c] = df[c].astype(str)
    out["Year"] = df["Year"]
    if TARGET in df:
        out[TARGET] = (df[TARGET] == "Yes").astype(int)
    return out


def time_split(features: pd.DataFrame, valid_year: int = 1995, test_year: int = 1996):
    """Train on earlier years, tune on the next one, test on the last: the order claims actually arrive in."""
    train = features[features["Year"] < valid_year]
    valid = features[features["Year"] == valid_year]
    test = features[features["Year"] >= test_year]
    return train, valid, test
