"""
Stage 2 of the pipeline: turn the train/test CSVs into model-ready
numeric arrays, and save the fitted preprocessor so the exact same
transformation can be replayed on new employee data at prediction time
(this is what prevents train/serve skew).

Numeric columns are scaled; categorical columns are one-hot encoded
(not label-encoded) since fields like Department or JobRole have no
natural order — label encoding would falsely imply e.g. Sales < R&D.
"""

import sys
import logging
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from employee_attrition.exception import AttritionException
from employee_attrition.entity.config_entity import DataTransformationConfig
from employee_attrition.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
)
from employee_attrition.utils import save_object


class DataTransformation:
    def __init__(self, config: DataTransformationConfig = DataTransformationConfig()):
        self.config = config

    def get_preprocessor(self, numeric_cols, categorical_cols) -> ColumnTransformer:
        numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
        categorical_pipeline = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))]
        )
        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_cols),
                ("cat", categorical_pipeline, categorical_cols),
            ]
        )

    def initiate_data_transformation(
        self, ingestion_artifact: DataIngestionArtifact
    ) -> DataTransformationArtifact:
        logging.info("Starting data transformation")
        try:
            train_df = pd.read_csv(ingestion_artifact.train_data_path)
            test_df = pd.read_csv(ingestion_artifact.test_data_path)

            train_df = train_df.drop(columns=list(self.config.drop_columns), errors="ignore")
            test_df = test_df.drop(columns=list(self.config.drop_columns), errors="ignore")

            target_col = self.config.target_column
            y_train = train_df[target_col].map({"Yes": 1, "No": 0})
            y_test = test_df[target_col].map({"Yes": 1, "No": 0})

            X_train = train_df.drop(columns=[target_col])
            X_test = test_df.drop(columns=[target_col])

            numeric_cols = X_train.select_dtypes(exclude="object").columns.tolist()
            categorical_cols = X_train.select_dtypes(include="object").columns.tolist()
            logging.info(f"Numeric columns: {numeric_cols}")
            logging.info(f"Categorical columns: {categorical_cols}")

            preprocessor = self.get_preprocessor(numeric_cols, categorical_cols)

            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            # densify if sparse (OneHotEncoder can return a sparse matrix)
            if hasattr(X_train_arr, "toarray"):
                X_train_arr = X_train_arr.toarray()
            if hasattr(X_test_arr, "toarray"):
                X_test_arr = X_test_arr.toarray()

            train_array = np.c_[X_train_arr, y_train.to_numpy()]
            test_array = np.c_[X_test_arr, y_test.to_numpy()]

            save_object(self.config.preprocessor_path, preprocessor)
            logging.info(f"Saved preprocessor to {self.config.preprocessor_path}")

            return DataTransformationArtifact(
                preprocessor_path=self.config.preprocessor_path,
                train_array=train_array,
                test_array=test_array,
            )
        except Exception as e:
            raise AttritionException(e, sys)
