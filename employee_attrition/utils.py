"""
Small reusable helpers used across components — saving/loading
Python objects (models, preprocessors) and writing evaluation
metrics to disk in a consistent way.
"""

import os
import sys
import json
import dill

from employee_attrition.exception import AttritionException


def save_object(file_path: str, obj) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            dill.dump(obj, f)
    except Exception as e:
        raise AttritionException(e, sys)


def load_object(file_path: str):
    try:
        with open(file_path, "rb") as f:
            return dill.load(f)
    except Exception as e:
        raise AttritionException(e, sys)


def save_json(file_path: str, data: dict) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        raise AttritionException(e, sys)
