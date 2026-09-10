"""
Streamlit app — the "application" layer of the project.

Loads the trained model + preprocessor (via PredictPipeline) and lets
a user enter one employee's details to get an attrition risk score.
Run with: streamlit run app.py
"""

import json
import streamlit as st
import pandas as pd

from employee_attrition.pipeline.predict_pipeline import PredictPipeline, EmployeeData
from employee_attrition import config

st.set_page_config(page_title="Employee Attrition Predictor", page_icon="📊", layout="centered")

st.title("📊 Employee Attrition Prediction")
st.caption(
    "Enter an employee's details to estimate their risk of leaving the company. "
    "Powered by a Random Forest model trained on 4,300+ historical employee records (ROC-AUC 0.965)."
)

with st.form("employee_form"):
    st.subheader("Personal Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 18, 60, 30)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col3:
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

    st.subheader("Job Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        department = st.selectbox(
            "Department", ["Sales", "Research & Development", "Human Resources"]
        )
    with col2:
        job_role = st.selectbox(
            "Job Role",
            [
                "Sales Executive", "Research Scientist", "Laboratory Technician",
                "Manufacturing Director", "Healthcare Representative", "Manager",
                "Sales Representative", "Research Director", "Human Resources",
            ],
        )
    with col3:
        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])

    col1, col2, col3 = st.columns(3)
    with col1:
        business_travel = st.selectbox(
            "Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
        )
    with col2:
        education = st.selectbox("Education Level (1=Below College, 5=Doctor)", [1, 2, 3, 4, 5])
    with col3:
        education_field = st.selectbox(
            "Education Field",
            ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"],
        )

    st.subheader("Compensation & Tenure")
    col1, col2, col3 = st.columns(3)
    with col1:
        monthly_income = st.number_input("Monthly Income", 10090, 199990, 50000, step=1000)
    with col2:
        percent_salary_hike = st.slider("Last Salary Hike (%)", 11, 25, 15)
    with col3:
        stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3])

    col1, col2, col3 = st.columns(3)
    with col1:
        total_working_years = st.number_input("Total Working Years", 0, 40, 5)
    with col2:
        years_at_company = st.number_input("Years at Company", 0, 40, 3)
    with col3:
        num_companies_worked = st.number_input("Num Companies Worked", 0, 9, 1)

    col1, col2, col3 = st.columns(3)
    with col1:
        years_since_promotion = st.number_input("Years Since Last Promotion", 0, 15, 1)
    with col2:
        years_with_manager = st.number_input("Years With Current Manager", 0, 17, 2)
    with col3:
        training_times = st.number_input("Training Times Last Year", 0, 6, 2)

    col1, col2 = st.columns(2)
    with col1:
        distance_from_home = st.number_input("Distance From Home (km)", 1, 29, 5)
    with col2:
        performance_rating = st.selectbox("Performance Rating", [3, 4])

    st.subheader("Satisfaction & Involvement")
    col1, col2, col3 = st.columns(3)
    with col1:
        env_satisfaction = st.selectbox("Environment Satisfaction (1-4)", [1.0, 2.0, 3.0, 4.0], index=2)
    with col2:
        job_satisfaction = st.selectbox("Job Satisfaction (1-4)", [1.0, 2.0, 3.0, 4.0], index=2)
    with col3:
        work_life_balance = st.selectbox("Work-Life Balance (1-4)", [1.0, 2.0, 3.0, 4.0], index=2)

    job_involvement = st.selectbox("Job Involvement (1-4)", [1, 2, 3, 4], index=2)

    submitted = st.form_submit_button("Predict Attrition Risk", use_container_width=True)

if submitted:
    employee = EmployeeData(
        Age=age,
        BusinessTravel=business_travel,
        Department=department,
        DistanceFromHome=distance_from_home,
        Education=education,
        EducationField=education_field,
        Gender=gender,
        JobLevel=job_level,
        JobRole=job_role,
        MaritalStatus=marital_status,
        MonthlyIncome=monthly_income,
        NumCompaniesWorked=num_companies_worked,
        PercentSalaryHike=percent_salary_hike,
        StockOptionLevel=stock_option_level,
        TotalWorkingYears=total_working_years,
        TrainingTimesLastYear=training_times,
        YearsAtCompany=years_at_company,
        YearsSinceLastPromotion=years_since_promotion,
        YearsWithCurrManager=years_with_manager,
        EnvironmentSatisfaction=env_satisfaction,
        JobSatisfaction=job_satisfaction,
        WorkLifeBalance=work_life_balance,
        JobInvolvement=job_involvement,
        PerformanceRating=performance_rating,
    )

    try:
        pipeline = PredictPipeline()
        pred, proba = pipeline.predict(employee.to_dataframe())

        risk_pct = proba[0] * 100
        st.divider()

        if pred[0] == 1:
            st.error(f"⚠️ High Attrition Risk — {risk_pct:.1f}% probability of leaving")
        else:
            st.success(f"✅ Low Attrition Risk — {risk_pct:.1f}% probability of leaving")

        st.progress(min(int(risk_pct), 100))

        with st.expander("Model details"):
            with open(config.METRICS_PATH) as f:
                metrics = json.load(f)
            st.write(f"**Model used:** {metrics['best_model']}")
            st.json(metrics["all_models"][metrics["best_model"]])

    except FileNotFoundError:
        st.warning(
            "No trained model found. Run `python main.py` first to train and save the model, "
            "then restart this app."
        )

st.divider()
st.caption(
    "This tool is a decision-support aid, not a standalone HR decision. "
    "Always combine model output with manager context before acting on it."
)
