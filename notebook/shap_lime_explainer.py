import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import tensorflow as tf
from lime.lime_tabular import LimeTabularExplainer

X = pd.read_csv('X_test.csv')
X_train_summary = shap.utils.sample(X, 100)

models = {
    "XGBoost": joblib.load('xgboost_model.pkl'),
    "RandomForest": joblib.load('random_forest_model.pkl'),
    "LightGBM": joblib.load('lightgbm_model.pkl'),
    "Tabular_DL": tf.keras.models.load_model('tabular_dl_model.keras')
}

i = 0

for name, model in models.items():
    print(f"\nProcessing: {name}")

    if name == "Tabular_DL":
        explainer_shap = shap.GradientExplainer(model, X_train_summary.values)
        shap_values_all = explainer_shap.shap_values(X.values[:100])

        if isinstance(shap_values_all, list):
            shap_values_plot = shap_values_all[0]
        else:
            shap_values_plot = shap_values_all

        plt.figure()
        shap.summary_plot(shap_values_plot, X.iloc[:100], show=False)
        plt.savefig(f"{name}_shap_summary.png", bbox_inches='tight')
        plt.close()
    else:
        explainer_shap = shap.Explainer(model, X_train_summary)
        shap_values_obj = explainer_shap(X)

        plt.figure()
        shap.plots.beeswarm(shap_values_obj, show=False)
        plt.savefig(f"{name}_shap_summary.png", bbox_inches='tight')
        plt.close()

        plt.figure()
        shap.plots.waterfall(shap_values_obj[i], show=False)
        plt.savefig(f"{name}_shap_customer.png", bbox_inches='tight')
        plt.close()

    if hasattr(model, "predict_proba"):
        predict_fn = model.predict_proba
        mode = 'classification'
    else:
        predict_fn = model.predict
        mode = 'regression'

    explainer_lime = LimeTabularExplainer(
        training_data=X.values,
        feature_names=X.columns.tolist(),
        mode=mode,
        discretize_continuous=True
    )

    exp = explainer_lime.explain_instance(
        X.iloc[i].values,
        predict_fn,
        num_features=10
    )

    fig = exp.as_pyplot_figure()
    plt.title(f"{name} - LIME Explanation (Customer {i})")
    plt.savefig(f"{name}_lime.png", bbox_inches='tight')
    plt.close()

    print(f"Completed {name}. Files saved.")