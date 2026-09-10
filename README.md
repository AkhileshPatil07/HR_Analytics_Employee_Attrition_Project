# 📊 Employee Attrition Prediction

A machine learning project that predicts whether an employee is likely to leave a company, based on their profile — demographics, job role, compensation, tenure, and satisfaction scores. The project follows a **modular ML pipeline architecture**, separating data ingestion, transformation, model training, and prediction into independent, testable components.


---

## 🚀 Project Overview

Organizations lose significant time and money replacing employees who leave unexpectedly. This project builds a model that scores an individual employee's attrition risk, so HR teams can intervene proactively rather than reactively.

### 🔄 Machine Learning Workflow

```
Dataset (4,327 employees)
   ↓
Data Ingestion (stratified train/test split)
   ↓
Data Transformation (scaling + one-hot encoding, saved as preprocessor.pkl)
   ↓
Model Training (Logistic Regression vs Random Forest, best model selected on ROC-AUC)
   ↓
Model Artifact (model.pkl, metrics.json, feature_importance.csv)
   ↓
Prediction Pipeline
   ↓
Streamlit Web Application
```

---

## ✨ Key Features

- 🎯 Employee attrition risk prediction (probability, not just yes/no)
- 🔄 Modular ML pipeline (ingestion → transformation → training → prediction)
- ⚖️ Class-imbalance handling (only 16.2% of employees left — accuracy alone is misleading)
- 📈 Two models compared honestly: Logistic Regression (baseline) vs Random Forest (final)
- 🔍 Feature importance output — explains *why* the model flags someone as high-risk
- 📝 Logging system for tracking pipeline execution
- ⚠️ Custom exception handling with file/line-level error tracing
- 🧪 Model checking utility to verify saved artifacts before deployment
- 🌐 Streamlit web app for interactive, single-employee predictions

---

## 📈 Results

| Model | ROC-AUC | Recall (Left) | Precision (Left) |
|---|---|---|---|
| Logistic Regression | 0.790 | 0.71 | 0.31 |
| **Random Forest (selected)** | **0.965** | **0.78** | **0.73** |

**Top 5 attrition drivers** (Random Forest feature importance): Age, Total Working Years, Years at Company, Years with Current Manager, Monthly Income.

The model is tuned to prioritize **recall** over raw accuracy — missing an actual leaver (false negative) costs HR more than a false alarm, since a flagged employee can simply be double-checked, but an employee the model never flags gets no intervention at all.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| ML & Data Processing | Pandas, NumPy, Scikit-learn |
| Application | Streamlit |
| Serialization | dill |
| Dev Tools | Git, GitHub, Virtual Environment |

---

## 📂 Project Structure

```
Employee_Attrition_Prediction/
│
├── artifacts/                          # Generated: datasets, model, preprocessor, metrics
│   ├── train.csv
│   ├── test.csv
│   ├── preprocessor.pkl
│   ├── model.pkl
│   ├── metrics.json
│   └── feature_importance.csv
│
├── dataset/
│   └── HR_Analytics_Cleaned_Master.csv
│
├── employee_attrition/
│   │
│   ├── components/
│   │   ├── data_ingestion.py           # Load CSV, stratified train/test split
│   │   ├── data_transformation.py      # Scaling + one-hot encoding, saves preprocessor
│   │   └── model_trainer.py            # Trains & compares models, saves the best one
│   │
│   ├── entity/
│   │   ├── config_entity.py            # Dataclasses: what each stage needs
│   │   └── artifact_entity.py          # Dataclasses: what each stage produces
│   │
│   ├── pipeline/
│   │   ├── train_pipeline.py           # Orchestrates ingestion → transformation → training
│   │   └── predict_pipeline.py         # Loads saved model, scores new employee data
│   │
│   ├── __init__.py
│   ├── config.py                       # All paths & constants in one place
│   ├── exception.py                    # Custom exception with file/line tracing
│   ├── logger.py                       # Logging configuration
│   └── utils.py                        # save_object / load_object / save_json helpers
│
├── .gitignore
├── README.md
├── app.py                              # Streamlit application entry point
├── check_model.py                      # Verify saved model & preprocessor
├── main.py                             # Training pipeline entry point
└── requirements.txt
```

---

## 🔍 Project Modules

### 1. Components
Core data-processing and modeling logic: ingestion, transformation, and training — each independently runnable and testable.

### 2. Entity
Dataclasses that separate *configuration* (what a stage needs) from *artifacts* (what a stage produces), so components don't depend on each other's internals.

### 3. Pipeline
`train_pipeline.py` wires the three components together for training. `predict_pipeline.py` loads the saved model and preprocessor to score new data — the same transformation used in training is replayed exactly, preventing train/serve skew.

### 4. Configuration
`config.py` centralizes every file path and hyperparameter. Nothing else in the codebase hardcodes a path.

### 5. Logging
`logger.py` writes timestamped logs to `logs/` and the console for every pipeline run — useful for debugging which stage failed and why.

### 6. Exception Handling
`exception.py` wraps errors with the exact file and line number where they occurred, instead of a generic traceback.

### 7. Utilities
`utils.py` provides `save_object` / `load_object` (model & preprocessor persistence via `dill`) and `save_json` (metrics logging).

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clonehttps://github.com/Akhi leshPatil07/HR_Analytics_Employee_Attrition_Project
cd Employee_Attrition_Prediction
```

### 2. Create a virtual environment

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Train the model
```bash
python main.py
```
This runs the full pipeline (ingestion → transformation → training) and saves `model.pkl`, `preprocessor.pkl`, and `metrics.json` to `artifacts/`.

### Verify the saved model
```bash
python check_model.py
```

### Run the web application
```bash
streamlit run app.py
```
Open the URL shown in the terminal, fill in an employee's details, and get an attrition risk score.

---

## 📊 Prediction Process

```
Employee Details (form input)
        ↓
Same Preprocessor Used in Training
        ↓
Trained Random Forest Model
        ↓
Attrition Probability + Risk Label
```

---

## 📦 Model Artifacts

`artifacts/` stores everything the pipeline generates:
- `train.csv` / `test.csv` — the stratified split
- `preprocessor.pkl` — fitted `ColumnTransformer` (scaler + one-hot encoder)
- `model.pkl` — the selected model (Random Forest)
- `metrics.json` — ROC-AUC, recall, precision for both candidate models
- `feature_importance.csv` — raw feature importances from the Random Forest

---

## 🎯 Project Objectives

- Build an end-to-end, production-style ML prediction system (not just a notebook).
- Handle real-world data challenges: class imbalance, categorical encoding, train/serve consistency.
- Implement a modular, maintainable architecture separating config, data, and logic.
- Make the model's reasoning visible via feature importance, not just its output.
- Deploy the prediction system through a usable web application.

---

## 🔮 Future Improvements

- Deploy the application to a cloud platform (Streamlit Cloud / Render).
- Add hyperparameter tuning (GridSearchCV / Optuna).
- Add SHAP values for per-prediction explainability.
- Add API-based prediction using FastAPI or Flask.
- Add model performance monitoring / drift detection.
- Add CI/CD for automated retraining and deployment.

---

## 👨‍💻 Author

**Akhilesh Patil**

Data Analyst / Data Scientist with hands-on experience in Python, SQL, Machine Learning, and Data Analysis.

---

## 📄 License

This project is distributed under the MIT License.
