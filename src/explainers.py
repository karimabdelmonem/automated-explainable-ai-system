from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer


def build_shap_explainer(model, background_df: pd.DataFrame | None = None):
    if background_df is not None and len(background_df) > 0:
        sample_size = min(len(background_df), 100)
        background = shap.utils.sample(background_df, sample_size, random_state=42)
        return shap.Explainer(model, background)
    return shap.Explainer(model)


def plot_shap_waterfall(
    shap_explainer,
    transformed_row: pd.DataFrame,
    max_display: int = 10,
):
    shap_values = shap_explainer(transformed_row)
    single_explanation = shap_values[0]

    # For binary classifiers SHAP can return per-class attributions with shape
    # (n_features, 2). Select class index 1 (default probability) so waterfall
    # receives a 1D explanation for one outcome.
    if getattr(single_explanation, "values", None) is not None:
        values = single_explanation.values
        if hasattr(values, "ndim") and values.ndim == 2 and values.shape[1] == 2:
            single_explanation = shap_values[0, :, 1]

    fig = plt.figure()
    shap.plots.waterfall(single_explanation, max_display=max_display, show=False)
    return fig


def build_lime_explainer(background_df: pd.DataFrame, mode: str = "classification") -> LimeTabularExplainer:
    return LimeTabularExplainer(
        training_data=background_df.values,
        feature_names=background_df.columns.tolist(),
        mode=mode,
        discretize_continuous=True,
    )


def get_predict_fn(model) -> tuple[Callable, str]:
    if hasattr(model, "predict_proba"):
        return model.predict_proba, "classification"
    return model.predict, "regression"


def explain_with_lime(
    model,
    lime_explainer: LimeTabularExplainer,
    transformed_row: pd.DataFrame,
    num_features: int = 10,
):
    predict_fn, _ = get_predict_fn(model)
    exp = lime_explainer.explain_instance(
        transformed_row.iloc[0].values,
        predict_fn,
        num_features=num_features,
    )
    fig = exp.as_pyplot_figure()
    return exp, fig


def prediction_summary(model, transformed_row: pd.DataFrame) -> dict[str, float | int]:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(transformed_row)[0]
        pred = int(np.argmax(probs))
        return {"prediction": pred, "probability_class_1": float(probs[1]), "probability_class_0": float(probs[0])}

    pred = float(model.predict(transformed_row)[0])
    return {"prediction": pred}
