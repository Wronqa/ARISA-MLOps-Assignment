from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from ARISA_DSML.config import MODELS_DIR, PROCESSED_DATA_DIR

"""Run prediction on test data."""
from pathlib import Path
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
import shap
import joblib
import os
from ARISA_DSML.config import FIGURES_DIR, MODELS_DIR, target, PROCESSED_DATA_DIR, MODEL_NAME
from ARISA_DSML.resolve import get_model_by_alias
import mlflow
from mlflow.client import MlflowClient
import json
import nannyml as nml


app = typer.Typer()




def plot_shap(model:CatBoostClassifier, df_plot:pd.DataFrame)->None:
    """Plot model shapley overview plot."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_plot)

    shap.summary_plot(shap_values, df_plot, show=False)
    plt.savefig(FIGURES_DIR / "test_shap_overall.png")


def predict(model:CatBoostClassifier, df_pred:pd.DataFrame, params:dict, probs=False)->str|Path:
    """Do predictions on test data."""
    
    feature_columns = params.pop("feature_columns")
    
    preds = model.predict(df_pred[feature_columns])
    if probs:
        df_pred["predicted_probability"] = [p[1] for p in model.predict_proba(df_pred[feature_columns])]

    plot_shap(model, df_pred[feature_columns])
    df_pred[target] = preds
    preds_path = MODELS_DIR / "preds.csv"
    df_pred[[target]].to_csv(preds_path, index=False)

    
    return preds_path

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Performing inference for model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Inference complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
