"""
Entry point for running the full training pipeline end-to-end.

Usage:
    python main.py
"""

from employee_attrition.logger import logging  # noqa: F401 (configures logging on import)
from employee_attrition.pipeline.train_pipeline import TrainPipeline

if __name__ == "__main__":
    artifact = TrainPipeline().run_pipeline()
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"Best model : {artifact.best_model_name}")
    print(f"ROC-AUC    : {artifact.roc_auc:.3f}")
    print(f"Recall     : {artifact.recall:.3f}")
    print(f"Precision  : {artifact.precision:.3f}")
    print(f"Model saved at: {artifact.model_path}")
