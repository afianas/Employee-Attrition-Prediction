
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title='Employee Attrition Predictor', layout='centered')
st.title('Employee Attrition Predictor')

# ------------------------
# Load model, scaler, metadata
# ------------------------
@st.cache_resource
def load_artifacts():
    required_files = ["best_random_forest_attrition.pkl", "scaler.pkl",
                      "feature_names.pkl", "numeric_cols_to_scale.pkl"]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        return None, None, None, None, missing
    model = joblib.load("best_random_forest_attrition.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")               # list of column names in correct order
    numeric_cols_to_scale = joblib.load("numeric_cols_to_scale.pkl")  # list of numeric cols that were scaled
    return model, scaler, feature_names, numeric_cols_to_scale, None

model, scaler, feature_names, numeric_cols_to_scale, missing = load_artifacts()

if missing:
    st.error(f"Missing files: {missing}. Run the training notebook to generate them (model, scaler, feature_names, numeric_cols_to_scale).")
    st.stop()



# ------------------------
# Form for single input
# ------------------------
st.header("Enter Employee Details")

def user_input_form():
    data = {}
    data["Age"] = st.number_input("Age", 18, 65, 30)
    data["Gender"] = st.selectbox("Gender", ["Male", "Female"])
    data["Marital_Status"] = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    data["Department"] = st.selectbox("Department", ["Sales", "IT", "HR", "R&D", "Finance", "Marketing"])
    data["Job_Role"] = st.selectbox("Job Role", ["Sales Executive", "Software Engineer", "HR Specialist",
                                                 "Research Scientist", "Financial Analyst", "Marketing Specialist"])
    data["Job_Level"] = st.number_input("Job Level", 1, 5, 2)
    data["Monthly_Income"] = st.number_input("Monthly Income", 1000, 30000, 5000)
    data["Hourly_Rate"] = st.number_input("Hourly Rate", 10, 100, 40)
    data["Years_at_Company"] = st.number_input("Years at Company", 0, 40, 2)
    data["Years_in_Current_Role"] = st.number_input("Years in Current Role", 0, 20, 1)
    data["Years_Since_Last_Promotion"] = st.number_input("Years Since Last Promotion", 0, 20, 0)
    data["Work_Life_Balance"] = st.selectbox("Work Life Balance (1â€“4)", [1,2,3,4])
    data["Job_Satisfaction"] = st.selectbox("Job Satisfaction (1â€“5)", [1,2,3,4,5])
    data["Performance_Rating"] = st.selectbox("Performance Rating (1â€“4)", [1,2,3,4])
    data["Training_Hours_Last_Year"] = st.number_input("Training Hours Last Year", 0, 200, 20)
    data["Overtime"] = st.selectbox("Overtime", ["Yes", "No"])
    data["Project_Count"] = st.number_input("Project Count", 1, 20, 5)
    data["Average_Hours_Worked_Per_Week"] = st.number_input("Average Hours Worked Per Week", 30, 70, 45)
    data["Absenteeism"] = st.number_input("Absenteeism (days)", 0, 50, 5)
    data["Work_Environment_Satisfaction"] = st.selectbox("Work Environment Satisfaction (1â€“4)", [1,2,3,4])
    data["Relationship_with_Manager"] = st.selectbox("Relationship with Manager (1â€“4)", [1,2,3,4])
    data["Job_Involvement"] = st.selectbox("Job Involvement (1â€“4)", [1,2,3,4])
    data["Distance_From_Home"] = st.number_input("Distance From Home (km)", 1, 60, 10)
    data["Number_of_Companies_Worked"] = st.number_input("Number of Companies Worked", 0, 10, 1)
    return pd.DataFrame([data])

input_df = user_input_form()

st.subheader("Input preview")
st.dataframe(input_df.T)

# ------------------------
# Preprocess input to match training features
# ------------------------
# Categorical columns that were one-hot encoded during training
# We derive them by checking which feature_names contain those prefixes.
# But we will use a conservative list used previously: Gender, Marital_Status, Department, Job_Role, Overtime
cat_cols = ["Gender", "Marital_Status", "Department", "Job_Role", "Overtime"]

# One-hot encode user input (drop_first=True as in training)
df_encoded = pd.get_dummies(input_df, columns=[c for c in cat_cols if c in input_df.columns], drop_first=True)

# Ensure all training features are present in the correct order:
# 1) Add missing columns (with 0)
for c in feature_names:
    if c not in df_encoded.columns:
        df_encoded[c] = 0

# 2) Remove any extra columns (shouldn't happen since we added all training columns)
extracols = [c for c in df_encoded.columns if c not in feature_names]
if extracols:
    df_encoded = df_encoded.drop(columns=extracols)

# 3) Reorder columns to match training
df_encoded = df_encoded[feature_names]

# ------------------------
# Scale only the numeric columns that were scaled during training
# ------------------------
# numeric_cols_to_scale contains names that were present in the training X
# We need to safely apply scaler to these columns in our df (they are present)
num_scale_cols = [c for c in numeric_cols_to_scale if c in df_encoded.columns]

if len(num_scale_cols) > 0:
    # scaler expects the same feature names ordering as when it was fit.
    # During training you scaled only numeric_cols_to_scale in place (not the whole X).
    # So we will transform these numeric cols only:
    df_encoded[num_scale_cols] = scaler.transform(df_encoded[num_scale_cols])
else:
    st.warning("No numeric columns to scale were found in input - check numeric_cols_to_scale metadata.")

# ------------------------
# Prediction
# ------------------------
if st.button("Predict Attrition"):
    try:
        pred_prob = model.predict_proba(df_encoded)[:,1][0]
        pred = model.predict(df_encoded)[0]
        if pred == 1:
            st.error(f" Predicted: ATTRITION = YES   (probability = {pred_prob:.3f})")
        else:
            st.success(f" Predicted: ATTRITION = NO   (probability = {pred_prob:.3f})")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)
