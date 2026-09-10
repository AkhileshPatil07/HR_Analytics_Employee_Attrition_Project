"""
Stage 1 of the pipeline: read the raw CSV and produce a stratified
train/test split saved to artifacts/. Stratifying on Attrition matters
because only 16.2% of employees left — a plain random split can
under/over-represent leavers in the test set and skew every metric
downstream.
"""

import os
import sys
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from employee_attrition.exception import AttritionException
from employee_attrition.entity.config_entity import DataIngestionConfig
from employee_attrition.entity.artifact_entity import DataIngestionArtifact


class DataIngestion:
    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        self.config = config

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logging.info("Starting data ingestion")
        try:
            df = pd.read_csv(self.config.raw_data_path)
            logging.info(f"Loaded dataset with shape {df.shape}")

            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)

            train_df, test_df = train_test_split(
                df,
                test_size=self.config.test_size,
                stratify=df[self.config.target_column],
                random_state=self.config.random_state,
            )
            logging.info(
                f"Train shape: {train_df.shape}, Test shape: {test_df.shape}"
            )

            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)
            logging.info("Data ingestion completed successfully")

            return DataIngestionArtifact(
                train_data_path=self.config.train_data_path,
                test_data_path=self.config.test_data_path,
            )
        except Exception as e:
            raise AttritionException(e, sys)


if __name__ == "__main__":
    DataIngestion().initiate_data_ingestion()
