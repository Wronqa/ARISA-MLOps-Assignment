"""Functions for preprocessing the data."""

import os
from pathlib import Path
import re
import zipfile

from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


from loguru import logger
import pandas as pd

from ARISA_DSML.config import DATASET, PROCESSED_DATA_DIR, RAW_DATA_DIR


def get_raw_data(dataset_name:str=DATASET)->None:
    api = KaggleApi()
    api.authenticate()

    download_folder = Path(RAW_DATA_DIR)

    logger.info(f"RAW_DATA_DIR is: {RAW_DATA_DIR}")

    api.dataset_download_files(dataset_name, path=str(download_folder), unzip=True)




def preprocess_df(file:str|Path)->str|Path:

    df = pd.read_csv(file)
    df_ids = df.pop("id")

    smoking_status = [['formerly smoked', 'never smoked', 'smokes','Unknown']]  

    ordinal_encoder_smoking_status = create_ordinal_encoder(smoking_status)
    hot_encoder = OneHotEncoder(drop='first')

    preprocessor = ColumnTransformer(
    transformers=[
        ('ever_married', hot_encoder, ['ever_married']),
        ('work_type', hot_encoder, ['work_type']),
        ('gender', hot_encoder, ['gender']),
        ('Residence_type', hot_encoder, ['Residence_type']),
        ('smoking_status', ordinal_encoder_smoking_status, ['smoking_status'])
    ],
    remainder='passthrough'  
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])

    encoded_data = preprocessor.fit_transform(df)

    transformed_df = pd.DataFrame(encoded_data,columns=preprocessor.get_feature_names_out())

    df_train, df_test = train_test_split(transformed_df, test_size=0.2, random_state=42)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_path = PROCESSED_DATA_DIR / "train.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    logger.info(f"Train saved to {train_path}, Test saved to {test_path}")

    return train_path, test_path





def create_ordinal_encoder(categories_order):
    return OrdinalEncoder(categories=categories_order)



if __name__=="__main__":
    # get the train and test sets from default location
    logger.info("getting datasets")
    get_raw_data()

    # preprocess both sets
    #logger.info("preprocessing train.csv")
    #preprocess_df(RAW_DATA_DIR / "train.csv")
    #logger.info("preprocessing test.csv")
    #preprocess_df(RAW_DATA_DIR / "test.csv")