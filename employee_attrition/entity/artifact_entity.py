"""
Dataclasses representing what each pipeline stage *produces*.
The pipeline module passes these between stages instead of raw
tuples/paths, so it's always clear what a component returns.
"""

from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    train_data_path: str
    test_data_path: str


@dataclass
class DataTransformationArtifact:
    preprocessor_path: str
    train_array: object
    test_array: object


@dataclass
class ModelTrainerArtifact:
    model_path: str
    best_model_name: str
    roc_auc: float
    recall: float
    precision: float
