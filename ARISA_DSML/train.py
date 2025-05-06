from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
from pathlib import Path

from catboost import CatBoostClassifier, Pool, cv
import joblib
from loguru import logger
import mlflow
from mlflow.client import MlflowClient
import optuna
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import train_test_split

from ARISA_DSML.config import MODELS_DIR, PROCESSED_DATA_DIR

from ARISA_DSML.config import (
    FIGURES_DIR,
    MODEL_NAME,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    target,
)
from ARISA_DSML.helpers import get_git_commit_hash
import nannyml as nml

app = typer.Typer()




def run_hyperopt(X_train:pd.DataFrame, y_train:pd.DataFrame, test_size:float=0.25, n_trials:int=20, overwrite:bool=False)->str|Path:  # noqa: PLR0913
    """Run optuna hyperparameter tuning."""
    best_params_path = MODELS_DIR / "best_params.pkl"
    if not best_params_path.is_file() or overwrite:
        X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(X_train, y_train, test_size=test_size, random_state=42)

        def objective(trial:optuna.trial.Trial)->float:
            with mlflow.start_run(nested=True):
                params = {
                    "depth": trial.suggest_int("depth", 2, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3),
                    "iterations": trial.suggest_int("iterations", 50, 300),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-5, 100.0, log=True),
                    "bagging_temperature": trial.suggest_float("bagging_temperature", 0.01, 1),
                    "random_strength": trial.suggest_float("random_strength", 1e-5, 100.0, log=True),
                    "ignored_features": [0],
                }
                model = CatBoostClassifier(**params, verbose=0)
                model.fit(
                    X_train_opt,
                    y_train_opt,
                    eval_set=(X_val_opt, y_val_opt),
                    early_stopping_rounds=50,
                )
                mlflow.log_params(params)
                preds = model.predict(X_val_opt)
                probs = model.predict_proba(X_val_opt)

                f1 = f1_score(y_val_opt, preds)
                logloss = log_loss(y_val_opt, probs)
                mlflow.log_metric("f1", f1)
                mlflow.log_metric("logloss", logloss)

            return model.get_best_score()["validation"]["Logloss"]

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        joblib.dump(study.best_params, best_params_path)

        params = study.best_params
    else:
        params = joblib.load(best_params_path)
    logger.info("Best Parameters: " + str(params))
    return best_params_path




def train_cv(X_train:pd.DataFrame, y_train:pd.DataFrame, params:dict, eval_metric:str="F1", n:int=5)->str|Path:  # noqa: PLR0913
    """Do cross-validated training."""
    params["eval_metric"] = eval_metric
    params["loss_function"] = "Logloss"
    params["ignored_features"] = [0]  # ignore passengerid

    data = Pool(X_train, y_train)

    cv_results = cv(
        params=params,
        pool=data,
        fold_count=n,
        partition_random_seed=42,
        shuffle=True,
        plot=True,

    )

    cv_output_path = MODELS_DIR / "cv_results.csv"
    cv_results.to_csv(cv_output_path, index=False)

    return cv_output_path


def train(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    params: dict | None,
    artifact_name: str = "catboost_model_stroke",
    cv_results: pd.DataFrame | None = None,
) -> tuple[str | Path]:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    """Train model on full dataset without cross-validation."""

    
    if params is None:
        logger.info("Training model without tuned hyperparameters")
        params = {}

    with mlflow.start_run():
        # Rozszerz parametry tylko do logowania
        loggable_params = params.copy()
        loggable_params["feature_columns"] = list(X_train.columns)
        loggable_params["ignored_features"] = [0]

        # Usuń niedozwolone przed przekazaniem do CatBoostClassifier
        catboost_params = {
            k: v for k, v in params.items()
            if k not in ["feature_columns", "ignored_features"]
        }

        model = CatBoostClassifier(
            **catboost_params,
            verbose=True,
        )

        model.fit(
            X_train,
            y_train,
            verbose_eval=50,
            early_stopping_rounds=50,
            use_best_model=False,
            plot=True,
        )

        mlflow.log_params(loggable_params)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"{artifact_name}.cbm"
        model.save_model(model_path)
        mlflow.log_artifact(model_path)

        if cv_results is not None:
            cv_metric_mean = cv_results["test-F1-mean"].mean()
            mlflow.log_metric("f1_cv_mean", cv_metric_mean)

            fig1 = plot_error_scatter(
                df_plot=cv_results,
                name="Mean F1 Score",
                title="Cross-Validation (N=5) Mean F1 score with Error Bands",
                xtitle="Training Steps",
                ytitle="Performance Score",
                yaxis_range=[0.5, 1.0],
            )
            mlflow.log_figure(fig1, "test-F1-mean_vs_iterations.png")

            fig2 = plot_error_scatter(
                cv_results,
                x="iterations",
                y="test-Logloss-mean",
                err="test-Logloss-std",
                name="Mean logloss",
                title="Cross-Validation (N=5) Mean Logloss with Error Bands",
                xtitle="Training Steps",
                ytitle="Logloss",
            )
            mlflow.log_figure(fig2, "test-logloss-mean_vs_iterations.png")

        # MLflow model registry
        model_info = mlflow.catboost.log_model(
            cb_model=model,
            artifact_path="model",
            input_example=X_train,
            registered_model_name=MODEL_NAME,
        )

        client = MlflowClient(mlflow.get_tracking_uri())
        model_info = client.get_latest_versions(MODEL_NAME)[0]
        client.set_registered_model_alias(MODEL_NAME, "challenger", model_info.version)
        client.set_model_version_tag(
            name=model_info.name,
            version=model_info.version,
            key="git_sha",
            value=get_git_commit_hash(),
        )

        model_params_path = MODELS_DIR / "model_params.pkl"
        joblib.dump(loggable_params, model_params_path)

    return (model_path, model_params_path)


def plot_error_scatter(  # noqa: PLR0913
        df_plot:pd.DataFrame,
        x:str="iterations",
        y:str="test-F1-mean",
        err:str="test-F1-std",
        name:str="",
        title:str="",
        xtitle:str="",
        ytitle:str="",
        yaxis_range:list[float]|None=None,
    )->None:
    """Plot plotly scatter plots with error areas."""
    # Create figure
    fig = go.Figure()

    if not len(name):
        name = y

    # Add mean performance line
    fig.add_trace(
        go.Scatter(
            x=df_plot[x], y=df_plot[y], mode="lines", name=name, line={"color": "blue"},
        ),
    )

    # Add shaded error region
    fig.add_trace(
        go.Scatter(
            x=pd.concat([df_plot[y], df_plot[x][::-1]]),
            y=pd.concat([df_plot[y]+df_plot[err],
                         df_plot[y]-df_plot[err]]),
            fill="toself",
            fillcolor="rgba(0, 0, 255, 0.2)",
            line={"color":"rgba(255, 255, 255, 0)"},
            showlegend=False,
        ),
    )

    # Customize layout
    fig.update_layout(
        title=title,
        xaxis_title=xtitle,
        yaxis_title=ytitle,
        template="plotly_white",
    )

    if yaxis_range is not None:
        fig.update_layout(
            yaxis={"range": yaxis_range},
        )

    fig.show()
    fig.write_image(FIGURES_DIR / f"{y}_vs_{x}.png")
    return fig


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training some model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
