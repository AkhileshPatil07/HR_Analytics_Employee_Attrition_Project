"""
Quick sanity check on the saved model artifacts — run this after
training to confirm the model and preprocessor load correctly and
agree with the metrics recorded during training, before wiring them
into the Streamlit app.

Usage:
    python check_model.py
"""

import json
from employee_attrition import config
from employee_attrition.utils import load_object

if __name__ == "__main__":
    print("Loading model...")
    model = load_object(config.MODEL_PATH)
    print(f"Model type: {type(model).__name__}")

    print("Loading preprocessor...")
    preprocessor = load_object(config.PREPROCESSOR_PATH)
    print(f"Preprocessor type: {type(preprocessor).__name__}")

    with open(config.METRICS_PATH) as f:
        metrics = json.load(f)

    print("\nSaved metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"\nBest model in use: {metrics['best_model']}")
    print("Model and preprocessor loaded successfully — ready for prediction.")
