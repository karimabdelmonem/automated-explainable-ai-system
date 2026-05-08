from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# Expected model schema from Models/random_forest_model.pkl (27 features).
EXPECTED_FEATURES = [
    "age",
    "income",
    "loanamount",
    "creditscore",
    "monthsemployed",
    "numcreditlines",
    "interestrate",
    "loanterm",
    "dtiratio",
    "education",
    "hasmortgage",
    "hasdependents",
    "hascosigner",
    "employmenttype_Part-time",
    "employmenttype_Self-employed",
    "employmenttype_Unemployed",
    "maritalstatus_Married",
    "maritalstatus_Single",
    "loanpurpose_Business",
    "loanpurpose_Education",
    "loanpurpose_Home",
    "loanpurpose_Other",
    "loan_to_income",
    "monthly_income",
    "employment_ratio",
    "creditscore_band",
    "high_risk_flag",
]


@dataclass
class InputPreprocessor:
    expected_features: list[str]

    @classmethod
    def from_model_columns(cls, model_columns: list[str] | np.ndarray | None = None) -> "InputPreprocessor":
        if model_columns is None:
            return cls(expected_features=EXPECTED_FEATURES)
        return cls(expected_features=list(model_columns))

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.columns = (
            out.columns.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
        )
        return out

    @staticmethod
    def _prepare_base_features(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        cols_to_drop = ["loanid", "default", "model_prediction", "risk_score"]
        out = out.drop(columns=[c for c in cols_to_drop if c in out.columns], errors="ignore")

        for col in ["hasmortgage", "hasdependents", "hascosigner"]:
            if col in out.columns:
                out[col] = out[col].map({"yes": 1, "no": 0, "Yes": 1, "No": 0}).fillna(out[col])

        edu_order = ["High School", "Bachelor's", "Master's", "PhD"]
        if "education" in out.columns:
            mapping = {v: i for i, v in enumerate(edu_order)}
            out["education"] = out["education"].map(mapping).fillna(out["education"])

        for col in [
            "age",
            "income",
            "loanamount",
            "creditscore",
            "monthsemployed",
            "numcreditlines",
            "interestrate",
            "loanterm",
            "dtiratio",
            "education",
            "hasmortgage",
            "hasdependents",
            "hascosigner",
        ]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        categorical_columns = ["employmenttype", "maritalstatus", "loanpurpose"]
        present_categoricals = [c for c in categorical_columns if c in out.columns]
        if present_categoricals:
            out = pd.get_dummies(out, columns=present_categoricals, drop_first=True)

        if {"loanamount", "income"}.issubset(out.columns):
            out["loan_to_income"] = out["loanamount"] / (out["income"] + 1.0)
            out["monthly_income"] = out["income"] / 12.0

        if {"monthsemployed", "age"}.issubset(out.columns):
            out["employment_ratio"] = out["monthsemployed"] / (out["age"] * 12.0 + 1.0)

        if "creditscore" in out.columns:
            out["creditscore_band"] = (
                pd.cut(out["creditscore"], bins=[0, 580, 670, 740, 850], labels=[0, 1, 2, 3])
                .astype(float)
                .fillna(0)
                .astype(int)
            )

        if {"dtiratio", "creditscore"}.issubset(out.columns):
            out["high_risk_flag"] = ((out["dtiratio"] > 0.45) & (out["creditscore"] < 600)).astype(int)

        return out

    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = self._normalize_columns(raw_df)
        df = self._prepare_base_features(df)

        for col in self.expected_features:
            if col not in df.columns:
                df[col] = 0.0

        df = df[self.expected_features].copy()
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return df

    def transform_one(self, row: dict[str, Any]) -> pd.DataFrame:
        return self.transform(pd.DataFrame([row]))
