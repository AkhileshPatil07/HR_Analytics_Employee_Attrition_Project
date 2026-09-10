"""
Central place for every path and constant the pipeline uses.
Nothing else in the codebase hardcodes a path — if the dataset moves
or the artifacts folder gets renamed, this is the only file to touch.
"""

import os

# ---- Base paths ----
ROOT_DIR = os.getcwd()
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")

# ---- Source data ----
RAW_DATA_PATH = os.path.join(ROOT_DIR, "dataset", "HR_Analytics_Cleaned_Master.csv")

# ---- Data ingestion artifacts ----
TRAIN_DATA_PATH = os.path.join(ARTIFACTS_DIR, "train.csv")
TEST_DATA_PATH = os.path.join(ARTIFACTS_DIR, "test.csv")

# ---- Data transformation artifacts ----
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
TRAIN_ARRAY_PATH = os.path.join(ARTIFACTS_DIR, "train_array.npy")
TEST_ARRAY_PATH = os.path.join(ARTIFACTS_DIR, "test_array.npy")

# ---- Model artifacts ----
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")
FEATURE_IMPORTANCE_PATH = os.path.join(ARTIFACTS_DIR, "feature_importance.csv")

# ---- Target / split config ----
TARGET_COLUMN = "Attrition"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Columns dropped before modeling — ID column and the "_Label" columns,
# which are just human-readable duplicates of numeric columns already
# present (e.g. JobSatisfaction vs JobSatisfaction_Label). Keeping both
# would leak the same signal twice.
DROP_COLUMNS = [
    "EmployeeID",
    "Education_Label",
    "JobSatisfaction_Label",
    "EnvironmentSatisfaction_Label",
    "WorkLifeBalance_Label",
    "PerformanceRating_Label",
]
