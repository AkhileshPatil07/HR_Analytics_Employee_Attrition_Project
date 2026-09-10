"""
Wires the three components together: ingestion -> transformation ->
training. This is the single entry point main.py calls — none of the
individual components need to know about each other, they just pass
artifact objects forward.
"""

import sys
import logging

from employee_attrition.exception import AttritionException
from employee_attrition.components.data_ingestion import DataIngestion
from employee_attrition.components.data_transformation import DataTransformation
from employee_attrition.components.model_trainer import ModelTrainer


class TrainPipeline:
    def run_pipeline(self):
        try:
            logging.info("========== TRAINING PIPELINE STARTED ==========")

            ingestion_artifact = DataIngestion().initiate_data_ingestion()

            transformation_artifact = DataTransformation().initiate_data_transformation(
                ingestion_artifact
            )

            trainer_artifact = ModelTrainer().initiate_model_training(
                transformation_artifact
            )

            logging.info(
                f"Training complete. Best model: {trainer_artifact.best_model_name} | "
                f"ROC-AUC: {trainer_artifact.roc_auc:.3f} | "
                f"Recall: {trainer_artifact.recall:.3f} | "
                f"Precision: {trainer_artifact.precision:.3f}"
            )
            logging.info("========== TRAINING PIPELINE COMPLETED ==========")

            return trainer_artifact
        except Exception as e:
            raise AttritionException(e, sys)
