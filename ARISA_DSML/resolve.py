from ARISA_DSML.config import (
    MODEL_NAME,
)
import mlflow
from mlflow.client import MlflowClient
from loguru import logger


def get_model_by_alias(client, model_name:str=MODEL_NAME, alias:str="champion"):
    try:
        alias_mv = client.get_model_version_by_alias(model_name, alias)
    except Exception as e:
        if f"alias {alias} not found" in str(e):
            return None
        raise(e)
    return alias_mv

if __name__=="__main__":
    client = MlflowClient(mlflow.get_tracking_uri())
    champ_mv = get_model_by_alias(client)
    if champ_mv is None:
        chall_mv = get_model_by_alias(client, alias="challenger")
        if chall_mv is None:
            model_info = client.get_latest_versions(MODEL_NAME)[0]
            logger.info("Did not found champion or challenger, promoting newest model to champion.")
            client.set_registered_model_alias(MODEL_NAME, "champion", model_info.version)
        else:
            logger.info("Found challenger model with no champion, promoting challenger to champion.")
            client.delete_registered_model_alias(MODEL_NAME, "challenger")
            client.set_registered_model_alias(MODEL_NAME, "champion", chall_mv.version)

    chall_mv = get_model_by_alias(client, alias="challenger")

    if champ_mv and chall_mv:
        champ_run = client.get_run(champ_mv.run_id)
        f1_champ = champ_run.data.metrics["f1_cv_mean"]

        chall_run = client.get_run(chall_mv.run_id)
        f1_chall = chall_run.data.metrics["f1_cv_mean"]

        if f1_chall >= f1_champ:
            logger.info("Challenger model surpassed metric of current champion, promoting challenger to champion.")
            client.delete_registered_model_alias(MODEL_NAME, "challenger")
            client.set_registered_model_alias(MODEL_NAME, "champion", chall_mv.version)
        else:
            challenge_failed_exc = "Challenger model does not surpass metric of current champion, ending predict workflow."
            logger.error(challenge_failed_exc)
            raise(Exception(challenge_failed_exc))
    elif champ_mv and chall_mv is None:
        logger.info("No challenger to champion, continuing with prediction.")