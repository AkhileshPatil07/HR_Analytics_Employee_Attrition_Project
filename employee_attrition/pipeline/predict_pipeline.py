"""
Loads the saved preprocessor + model and scores new, unseen employee
records. This is what app.py calls behind the scenes — the same
preprocessor object fitted during training is reused here, so a new
employee's data goes through the identical transformation the model
was trained on.
"""

import sys
import pandas as pd

from employee_attrition.exception import AttritionException
from employee_attrition.utils import load_object
from employee_attrition import config


class PredictPipeline:
    def __init__(self):
        self.model = load_object(config.MODEL_PATH)
        self.preprocessor = load_object(config.PREPROCESSOR_PATH)

    def predict(self, employee_df: pd.DataFrame):
        try:
            features = self.preprocessor.transform(employee_df)
            if hasattr(features, "toarray"):
                features = features.toarray()

            proba = self.model.predict_proba(features)[:, 1]
            pred = self.model.predict(features)
            return pred, proba
        except Exception as e:
            raise AttritionException(e, sys)


class EmployeeData:
    """
    Wraps raw input fields (e.g. from a Streamlit form) into the
    single-row DataFrame the preprocessor expects.
    """

    def __init__(self, **kwargs):
        self.data = kwargs

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.data])
