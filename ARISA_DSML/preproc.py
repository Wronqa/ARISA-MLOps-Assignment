"""Functions for preprocessing the data."""

import os
from pathlib import Path
import re
import zipfile

from kaggle.api.kaggle_api_extended import KaggleApi
from loguru import logger
import pandas as pd

from ARISA_DSML.config import DATASET, PROCESSED_DATA_DIR, RAW_DATA_DIR


def get_raw_data(dataset_name:str=DATASET)->None:
    api = KaggleApi()
    api.authenticate()

    download_folder = Path(RAW_DATA_DIR)

    logger.info(f"RAW_DATA_DIR is: {RAW_DATA_DIR}")

    api.dataset_download_files(dataset_name, path=str(download_folder), unzip=True)






def extract_title(name:str)-> str|None:
    """Extract title from passenger name."""
    match = re.search(r",\s*([\w\s]+)\.", name)

    return match.group(1) if match else None


def preprocess_df(file:str|Path)->str|Path:
    """Preprocess datasets."""
    _, file_name = os.path.split(file)
    df_data = pd.read_csv(file)
    df_data = df_data.drop(columns=["Ticket"])

    df_data["Title"] = df_data["Name"].apply(extract_title)

    # pattern to match a letter followed by a number
    cabin_pattern = r"([A-Za-z]+)(\d+)"

    # run pattern on Cabin to extract all matches
    matches = df_data["Cabin"].str.extractall(cabin_pattern)
    matches = matches.reset_index()

    # create a new column for each letter and number matched
    result = matches.pivot(index="level_0", columns="match", values=[0, 1])
    result.columns = [f"{col[0]}_{col[1]}" for col in result.columns]

    # join to original train dataframe
    df_data = df_data.join(result[["0_0", "1_0"]])

    # fill nans
    df_data["1_0"] = df_data["1_0"].astype(float)
    df_data = df_data.fillna({"0_0": "N", "1_0": df_data["1_0"].mean()})
    df_data["1_0"] = df_data["1_0"].astype(int)

    # rename new columns and drop old ones
    df_data = df_data.rename(columns={"0_0": "Deck", "1_0": "CabinNumber"})
    df_data = df_data.drop(columns=["Cabin", "Name"], axis=1)
    df_data = df_data.fillna({"Embarked": "N", "Age": df_data["Age"].mean()})
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    outfile_path = PROCESSED_DATA_DIR / file_name
    df_data.to_csv(outfile_path, index=False)

    return outfile_path


if __name__=="__main__":
    # get the train and test sets from default location
    logger.info("getting datasets")
    get_raw_data()

    # preprocess both sets
    #logger.info("preprocessing train.csv")
    #preprocess_df(RAW_DATA_DIR / "train.csv")
    #logger.info("preprocessing test.csv")
    #preprocess_df(RAW_DATA_DIR / "test.csv")