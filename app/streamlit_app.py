import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ------------------------
# Page config
# ------------------------
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="wide"
)

# ------------------------
# Paths
# ------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

# ------------------------
# Load artifacts
# ------------------------
@st.cache_resource
def load_artifacts():
    files = [
        "best_random_forest_attrition.pkl",
        "scaler.pkl",
        "feature_names.pkl",
        "numeric_cols_to_scale.pkl"
    ]

    for f in files:
        if not (MODEL_DIR / f).exists():
            return None, None, None, None

    model = joblib.load(MODEL_DIR / "best_random_forest_attrition.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
    numeric_cols = joblib.load(MODEL_DIR / "numeric_cols_to_scale.pkl")

    return model, scaler, feature_names, numeric_cols


model, scaler, feature_names, numeric_cols_to_scale = load_artifacts()

if model is None:
    st.error("Model files not found. Please train the model first.")
    st.stop()

# ------------------------
# Title section
# ------------------------
st.title("📊 Employee Attrition Prediction System")
st.markdown(
    """
    This application predicts whether an employee is **likely to leave the organization**
    based on demographic, job-related, and performance features.
    """
)

st.divider()

# ------------------------
# Sidebar inputs
# ------------------------
st.sidebar.header("🧾 Employee Details")

with st.sidebar:
    st.subheader("Personal Info")
    Age = st.slider("Age", 18, 65, 30)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Marital_Status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    Distance_From_Home = st.slider("Distance From Home (km)", 1, 60, 10)

    st.subheader("Job Info")
    Department = st.selectbox(
        "Department", ["Sales", "IT", "HR", "R&D", "Finance", "Marketing"]
    )
    Job_Role = st.selectbox(
        "Job Role",
        [
            "Sales Executive",
            "Software Engineer",
            "HR Specialist",
            "Research Scientist",
            "Financial Analyst",
            "Marketing Specialist",
        ],
    )
    Job_Level = st.slider("Job Level", 1, 5, 2)
    Monthly_Income = st.number_input("Monthly Income", 1000, 30000, 5000)
    Hourly_Rate = st.slider("Hourly Rate", 10, 100, 40)
    Overtime = st.selectbox("Overtime", ["Yes", "No"])

    st.subheader("Experience")
    Years_at_Company = st.slider("Years at Company", 0, 40, 2)
    Years_in_Current_Role = st.slider("Years in Current Role", 0, 20, 1)
    Years_Since_Last_Promotion = st.slider("Years Since Last Promotion", 0, 20, 0)
    Number_of_Companies_Worked = st.slider("Companies Worked", 0, 10, 1)

    st.subheader("Work Metrics")
    Work_Life_Balance = st.selectbox("Work Life Balance (1–4)", [1, 2, 3, 4])
    Job_Satisfaction = st.selectbox("Job Satisfaction (1–5)", [1, 2, 3, 4, 5])
    Work_Environment_Satisfaction = st.selectbox(
        "Work Environment Satisfaction (1–4)", [1, 2, 3, 4]
    )
    Relationship_with_Manager = st.selectbox(
        "Relationship with Manager (1–4)", [1, 2, 3, 4]
    )
    Job_Involvement = st.selectbox("Job Involvement (1–4)", [1, 2, 3, 4])
    Performance_Rating = st.selectbox("Performance Rating (1–4)", [1, 2, 3, 4])
    Training_Hours_Last_Year = st.slider("Training Hours Last Year", 0, 200, 20)
    Project_Count = st.slider("Project Count", 1, 20, 5)
    Average_Hours_Worked_Per_Week = st.slider("Avg Hours / Week", 30, 70, 45)
    Absenteeism = st.slider("Absenteeism (days)", 0, 50, 5)

# ------------------------
# Build input dataframe
# ------------------------
input_df = pd.DataFrame([{
    "Age": Age,
    "Gender": Gender,
    "Marital_Status": Marital_Status,
    "Department": Department,
    "Job_Role": Job_Role,
    "Job_Level": Job_Level,
    "Monthly_Income": Monthly_Income,
    "Hourly_Rate": Hourly_Rate,
    "Years_at_Company": Years_at_Company,
    "Years_in_Current_Role": Years_in_Current_Role,
    "Years_Since_Last_Promotion": Years_Since_Last_Promotion,
    "Work_Life_Balance": Work_Life_Balance,
    "Job_Satisfaction": Job_Satisfaction,
    "Performance_Rating": Performance_Rating,
    "Training_Hours_Last_Year": Training_Hours_Last_Year,
    "Overtime": Overtime,
    "Project_Count": Project_Count,
    "Average_Hours_Worked_Per_Week": Average_Hours_Worked_Per_Week,
    "Absenteeism": Absenteeism,
    "Work_Environment_Satisfaction": Work_Environment_Satisfaction,
    "Relationship_with_Manager": Relationship_with_Manager,
    "Job_Involvement": Job_Involvement,
    "Distance_From_Home": Distance_From_Home,
    "Number_of_Companies_Worked": Number_of_Companies_Worked
}])

# ------------------------
# Preprocessing
# ------------------------
cat_cols = ["Gender", "Marital_Status", "Department", "Job_Role", "Overtime"]
df_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)

for col in feature_names:
    if col not in df_encoded.columns:
        df_encoded[col] = 0

df_encoded = df_encoded[feature_names]

scale_cols = [c for c in numeric_cols_to_scale if c in df_encoded.columns]
df_encoded[scale_cols] = scaler.transform(df_encoded[scale_cols])

# ------------------------
# Prediction section
# ------------------------
st.divider()
st.subheader("🔮 Prediction Result")

if st.button("🚀 Predict Attrition", use_container_width=True):
    prob = model.predict_proba(df_encoded)[0][1]
    pred = model.predict(df_encoded)[0]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Attrition Probability", f"{prob:.2%}")

    with col2:
        if pred == 1:
            st.error("⚠️ High Risk of Attrition")
        else:
            st.success("✅ Low Risk of Attrition")

st.divider()

st.caption("Built with Machine Learning & Streamlit")
