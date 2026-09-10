"""
Dataclasses that bundle up the config each pipeline stage needs.
Keeping these separate from the components themselves means a
component's __init__ signature never has to change just because a
new path or hyperparameter gets added — it just reads from its config
object.
"""

from dataclasses import dataclass
from employee_attrition import config


@dataclass
class DataIngestionConfig:
    raw_data_path: str = config.RAW_DATA_PATH
    train_data_path: str = config.TRAIN_DATA_PATH
    test_data_path: str = config.TEST_DATA_PATH
    test_size: float = config.TEST_SIZE
    random_state: int = config.RANDOM_STATE
    target_column: str = config.TARGET_COLUMN


@dataclass
class DataTransformationConfig:
    preprocessor_path: str = config.PREPROCESSOR_PATH
    target_column: str = config.TARGET_COLUMN
    drop_columns: tuple = tuple(config.DROP_COLUMNS)


@dataclass
class ModelTrainerConfig:
    model_path: str = config.MODEL_PATH
    metrics_path: str = config.METRICS_PATH
    feature_importance_path: str = config.FEATURE_IMPORTANCE_PATH
    random_state: int = config.RANDOM_STATE
