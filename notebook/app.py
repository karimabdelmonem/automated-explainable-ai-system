from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn import __version__ as sklearn_version

from src.explainers import (
    build_lime_explainer,
    build_shap_explainer,
    explain_with_lime,
    prediction_summary,
    plot_shap_waterfall,
)
from src.preprocessing import InputPreprocessor


MODEL_PATH = Path("Models/random_forest_model.pkl")
TRAIN_FEATURES_PATH = Path("data/processed/X_train.parquet")
TRAINED_SKLEARN_VERSION = "1.6.1"


def render_version_warning() -> None:
    if sklearn_version != TRAINED_SKLEARN_VERSION:
        st.warning(
            (
                f"scikit-learn version mismatch detected: trained={TRAINED_SKLEARN_VERSION}, "
                f"current={sklearn_version}. For reproducible predictions, install "
                f"`scikit-learn=={TRAINED_SKLEARN_VERSION}`."
            )
        )


@st.cache_resource
def load_model():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(MODEL_PATH)


@st.cache_data
def load_background_data(expected_features: list[str]) -> pd.DataFrame:
    # Prefer training data if parquet dependencies are available.
    try:
        df = pd.read_parquet(TRAIN_FEATURES_PATH)
        missing = [c for c in expected_features if c not in df.columns]
        for col in missing:
            df[col] = 0.0
        return df[expected_features].fillna(0.0)
    except Exception:
        return pd.DataFrame([{c: 0.0 for c in expected_features}])


def input_form() -> dict:
    st.subheader("Loan Applicant Input")

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        income = st.number_input("Income", min_value=0.0, value=55000.0, step=1000.0)
        loan_amount = st.number_input("Loan Amount", min_value=0.0, value=15000.0, step=500.0)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=680)
    with c2:
        months_employed = st.number_input("Months Employed", min_value=0, value=60)
        num_credit_lines = st.number_input("Num Credit Lines", min_value=0, value=5)
        interest_rate = st.number_input("Interest Rate", min_value=0.0, max_value=1.0, value=0.12, step=0.01)
        loan_term = st.number_input("Loan Term (months)", min_value=6, value=36)
    with c3:
        dti_ratio = st.number_input("DTI Ratio", min_value=0.0, max_value=2.0, value=0.35, step=0.01)
        education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
        employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
        marital_status = st.selectbox("Marital Status", ["Divorced", "Married", "Single"])

    loan_purpose = st.selectbox("Loan Purpose", ["Auto", "Business", "Education", "Home", "Other"])

    b1, b2, b3 = st.columns(3)
    with b1:
        has_mortgage = st.selectbox("Has Mortgage", ["No", "Yes"])
    with b2:
        has_dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    with b3:
        has_cosigner = st.selectbox("Has Co-signer", ["No", "Yes"])

    return {
        "age": age,
        "income": income,
        "loanamount": loan_amount,
        "creditscore": credit_score,
        "monthsemployed": months_employed,
        "numcreditlines": num_credit_lines,
        "interestrate": interest_rate,
        "loanterm": loan_term,
        "dtiratio": dti_ratio,
        "education": education,
        "employmenttype": employment_type,
        "maritalstatus": marital_status,
        "loanpurpose": loan_purpose,
        "hasmortgage": has_mortgage,
        "hasdependents": has_dependents,
        "hascosigner": has_cosigner,
    }


def main() -> None:
    st.set_page_config(page_title="Explainable Loan Default Predictor", layout="wide")
    st.title("Automated Explainable AI - Random Forest")

    if not MODEL_PATH.exists():
        st.error(f"Model not found at `{MODEL_PATH}`.")
        st.stop()

    render_version_warning()
    model = load_model()
    expected_features = list(getattr(model, "feature_names_in_", []))
    if not expected_features:
        st.error("Model does not expose `feature_names_in_`; cannot safely align input schema.")
        st.stop()

    preprocessor = InputPreprocessor.from_model_columns(expected_features)
    background_df = load_background_data(expected_features)
    shap_explainer = build_shap_explainer(model, background_df)
    lime_explainer = build_lime_explainer(background_df, mode="classification")

    row = input_form()
    if st.button("Predict and Explain", type="primary"):
        transformed = preprocessor.transform_one(row)
        pred = prediction_summary(model, transformed)

        st.subheader("Prediction")
        st.write(
            {
                "Predicted Class": pred["prediction"],
                "Probability Default (class 1)": round(float(pred.get("probability_class_1", 0.0)), 4),
            }
        )

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("SHAP Explanation")
            shap_fig = plot_shap_waterfall(shap_explainer, transformed, max_display=10)
            st.pyplot(shap_fig, clear_figure=True)
        with col_right:
            st.subheader("LIME Explanation")
            _, lime_fig = explain_with_lime(model, lime_explainer, transformed, num_features=10)
            st.pyplot(lime_fig, clear_figure=True)

        with st.expander("Transformed Model Input (27 features)"):
            st.dataframe(transformed)


if __name__ == "__main__":
    main()
