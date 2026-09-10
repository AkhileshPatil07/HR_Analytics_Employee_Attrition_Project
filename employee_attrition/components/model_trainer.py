"""
Stage 3 of the pipeline: train Logistic Regression and Random Forest,
evaluate both on the held-out test set, and persist whichever model
wins on ROC-AUC (the metric that matters most given the 16.2%
attrition class imbalance — plain accuracy would reward a model that
just predicts "No" for everyone).
"""

import sys
import logging
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    classification_report,
)

from employee_attrition.exception import AttritionException
from employee_attrition.entity.config_entity import ModelTrainerConfig
from employee_attrition.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from employee_attrition.utils import save_object, save_json


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig = ModelTrainerConfig()):
        self.config = config

    def initiate_model_training(
        self, transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        logging.info("Starting model training")
        try:
            train_array = transformation_artifact.train_array
            test_array = transformation_artifact.test_array

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            candidates = {
                "LogisticRegression": LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=self.config.random_state,
                ),
                "RandomForest": RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=self.config.random_state,
                    n_jobs=-1,
                ),
            }

            results = {}
            fitted_models = {}

            for name, model in candidates.items():
                model.fit(X_train, y_train)
                proba = model.predict_proba(X_test)[:, 1]
                pred = model.predict(X_test)

                auc = roc_auc_score(y_test, proba)
                rec = recall_score(y_test, pred)
                prec = precision_score(y_test, pred)

                results[name] = {"roc_auc": auc, "recall": rec, "precision": prec}
                fitted_models[name] = model

                logging.info(f"{name} -> ROC-AUC: {auc:.3f}, Recall: {rec:.3f}, Precision: {prec:.3f}")
                logging.info(f"\n{classification_report(y_test, pred, target_names=['Stayed', 'Left'])}")

            best_name = max(results, key=lambda k: results[k]["roc_auc"])
            best_model = fitted_models[best_name]
            best_metrics = results[best_name]
            logging.info(f"Best model: {best_name} (ROC-AUC {best_metrics['roc_auc']:.3f})")

            save_object(self.config.model_path, best_model)
            save_json(self.config.metrics_path, {"all_models": results, "best_model": best_name})

            if hasattr(best_model, "feature_importances_"):
                pd.Series(best_model.feature_importances_).to_csv(
                    self.config.feature_importance_path, index_label="feature_index"
                )

            return ModelTrainerArtifact(
                model_path=self.config.model_path,
                best_model_name=best_name,
                roc_auc=best_metrics["roc_auc"],
                recall=best_metrics["recall"],
                precision=best_metrics["precision"],
            )
        except Exception as e:
            raise AttritionException(e, sys)
