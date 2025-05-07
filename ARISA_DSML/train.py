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



def run_hyperopt(X_train: pd.DataFrame, y_train: pd.DataFrame, test_size: float = 0.25, n_trials: int = 20, overwrite: bool = False) -> str | Path:  # noqa: PLR0913
    """Run optuna hyperparameter tuning."""
    best_params_path = MODELS_DIR / "best_params.pkl"
    if not best_params_path.is_file() or overwrite:
        X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(X_train, y_train, test_size=test_size, random_state=42)

        def objective(trial: optuna.trial.Trial) -> float:
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


def train_cv(X_train: pd.DataFrame, y_train: pd.DataFrame, params: dict, eval_metric: str = "F1", n: int = 5) -> str | Path:  # noqa: PLR0913
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
    artifact_name: str = "catboost_model_heart",
    cv_results: pd.DataFrame | None = None,
) -> tuple[Path, Path]:
    """Train model on full dataset, log with MLflow, and save artifacts."""
    if params is None:
        logger.info("Training model without tuned hyperparameters (using CatBoost defaults).")
        params = {}

    with mlflow.start_run() as run:
        logger.info(f"MLflow Run ID: {run.info.run_id} started.")

        mlflow_logged_params = params.copy()
        if hasattr(X_train, 'columns'):
            mlflow_logged_params["feature_columns"] = list(X_train.columns)

        catboost_training_params = {
            k: v for k, v in params.items()
            if k not in ["feature_columns"]
        }

        mlflow.log_params(mlflow_logged_params)
        logger.info(f"Logged parameters to MLflow: {mlflow_logged_params}")

        model = CatBoostClassifier(
            **catboost_training_params,
            verbose=0,
        )
        logger.info(f"Starting CatBoost model training with params: {catboost_training_params}")
        model.fit(
            X_train,
            y_train,
            verbose_eval=50,
            early_stopping_rounds=50,
            plot=False,
        )
        logger.info("Model training completed.")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_file_path = MODELS_DIR / f"{artifact_name}.cbm"
        model.save_model(str(model_file_path))
        mlflow.log_artifact(str(model_file_path), artifact_path="model_cbm_file")
        logger.info(f"Model saved to {model_file_path} and logged as MLflow artifact.")

        if cv_results is not None and not cv_results.empty:
            logger.info("Logging CV results and figures.")
            if "test-F1-mean" in cv_results.columns:
                cv_f1_mean_metric = cv_results["test-F1-mean"].mean()
                mlflow.log_metric("f1_cv_mean", cv_f1_mean_metric)
                logger.info(f"Logged CV F1 mean: {cv_f1_mean_metric:.4f}")

                fig1 = plot_error_scatter(
                    df_plot=cv_results,
                    name="Mean F1 Score",
                    title="Cross-Validation (N=5) Mean F1 score with Error Bands",
                    xtitle="Training Steps", ytitle="Performance Score",
                    yaxis_range=[0.5, 1.0],
                )
                if fig1:
                    mlflow.log_figure(fig1, "test-F1-mean_vs_iterations.png")

            if "test-Logloss-mean" in cv_results.columns and "iterations" in cv_results.columns:
                fig2 = plot_error_scatter(
                    df_plot=cv_results, x="iterations", y="test-Logloss-mean",
                    err="test-Logloss-std" if "test-Logloss-std" in cv_results.columns else None,
                    name="Mean logloss",
                    title="Cross-Validation (N=5) Mean Logloss with Error Bands",
                    xtitle="Training Steps", ytitle="Logloss",
                )
                if fig2:
                    mlflow.log_figure(fig2, "test-logloss-mean_vs_iterations.png")
                pass
        else:
            logger.info("No CV results to log.")

        logger.info("Logging model in MLflow format.")
        input_example = X_train.head(5) if isinstance(X_train, pd.DataFrame) else None

        registered_model_name_to_use = MODEL_NAME if 'MODEL_NAME' in globals() else None

        mlflow_model_info = mlflow.catboost.log_model(
            cb_model=model,
            artifact_path="mlflow_catboost_model",
            input_example=input_example,
            registered_model_name=registered_model_name_to_use,
        )
        logger.info(f"Model logged via mlflow.catboost.log_model at artifact path: {mlflow_model_info.artifact_path}")

        if registered_model_name_to_use:
            logger.info(f"Attempting to update registry for model: {registered_model_name_to_use}")
            client = MlflowClient()
            latest_versions = client.get_latest_versions(registered_model_name_to_use)
            if latest_versions:
                model_version_info = latest_versions[0]
                client.set_registered_model_alias(registered_model_name_to_use, "challenger", model_version_info.version)
                logger.info(f"Set alias 'challenger' for {registered_model_name_to_use} version {model_version_info.version}.")

                git_sha_val = get_git_commit_hash()  # Upewnij się, że get_git_commit_hash jest zdefiniowane
                if git_sha_val:
                    client.set_model_version_tag(
                        name=model_version_info.name, version=model_version_info.version,
                        key="git_sha", value=git_sha_val,
                    )
                    logger.info(f"Set tag 'git_sha': {git_sha_val} for model version.")
            else:
                logger.warning(f"No versions found for model '{registered_model_name_to_use}' in registry.")
        else:
            logger.info("Model not registered as no registered_model_name was provided/defined.")

        model_params_file_path = MODELS_DIR / "model_params.pkl"
        joblib.dump(mlflow_logged_params, model_params_file_path)
        logger.info(f"Logged parameters (mlflow_logged_params) saved locally to {model_params_file_path}")
        mlflow.log_artifact(str(model_params_file_path), artifact_path="run_configuration")

        """----------NannyML----------"""
        # Model monitoring initialization
        reference_df = X_train.copy()
        reference_df["prediction"] = model.predict(X_train)
        reference_df["predicted_probability"] = [p[1] for p in model.predict_proba(X_train)]
        reference_df[target] = y_train
        reference_df.drop(columns=["prediction", target, "predicted_probability"]).columns
        chunk_size = 50

        # univariate drift for features
        udc = nml.UnivariateDriftCalculator(
            column_names=X_train.columns,
            chunk_size=chunk_size,
        )
        udc.fit(reference_df.drop(columns=["prediction", target, "predicted_probability"]))

        # Confidence-based Performance Estimation for target
        estimator = nml.CBPE(
            problem_type="classification_binary",
            y_pred_proba="predicted_probability",
            y_pred="prediction",
            y_true=target,
            metrics=["roc_auc"],
            chunk_size=chunk_size,
        )
        estimator = estimator.fit(reference_df)

        store = nml.io.store.FilesystemStore(root_path=str(MODELS_DIR))
        store.store(udc, filename="udc.pkl")
        store.store(estimator, filename="estimator.pkl")

        mlflow.log_artifact(MODELS_DIR / "udc.pkl")
        mlflow.log_artifact(MODELS_DIR / "estimator.pkl")

    logger.info(f"MLflow Run ID: {run.info.run_id} completed.")
    return model_file_path, model_params_file_path


def plot_error_scatter(
    df_plot: pd.DataFrame,
    x: str = "iterations",
    y: str = "test-F1-mean",
    err: str = "test-F1-std",
    name: str = "",
    title: str = "",
    xtitle: str = "",
    ytitle: str = "",
    yaxis_range: list[float] | None = None,
) -> None:
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
            y=pd.concat([df_plot[y] + df_plot[err],
                         df_plot[y] - df_plot[err]]),
            fill="toself",
            fillcolor="rgba(0, 0, 255, 0.2)",
            line={"color": "rgba(255, 255, 255, 0)"},
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


def get_or_create_experiment(experiment_name: str):
    """Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters
    ----------
    - experiment_name (str): Name of the MLflow experiment.

    Returns
    -------
    - str: ID of the existing or newly created MLflow experiment.

    """
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id

    return mlflow.create_experiment(experiment_name)


if __name__ == "__main__":

    print('Comment for test starting github actions 7')

    df_train = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")

    y_train = df_train.pop(target)
    X_train = df_train

    experiment_id = get_or_create_experiment("stroke_prediction_hyperparam_tuning")
    mlflow.set_experiment(experiment_id=experiment_id)
    best_params_path = run_hyperopt(X_train, y_train)
    params = joblib.load(best_params_path)
    cv_output_path = train_cv(X_train, y_train, params)
    cv_results = pd.read_csv(cv_output_path)

    experiment_id = get_or_create_experiment("stroke_prediction_full_training")
    mlflow.set_experiment(experiment_id=experiment_id)
    model_path, model_params_path = train(X_train, y_train, params, cv_results=cv_results)

    cv_results = pd.read_csv(cv_output_path)
