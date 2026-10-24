import joblib
import pandas as pd
import streamlit as st

# Page setup
st.set_page_config(
    page_title="Diabetes Risk Prediction", page_icon="🩺", layout="centered"
)


# Load trained model (Updated to Logistic Regression)
@st.cache_resource
def load_model():
  return joblib.load("logistic_regression.joblib")


model = load_model()

# Header
st.title("🩺 Type 2 Diabetes Risk Prediction")
st.info("Enter your health and lifestyle details below to view your risk.")

# Inputs: Personal
st.header("Personal Information")
col1, col2 = st.columns(2)
with col1:
  gender = st.selectbox("Gender", ["Male", "Female"])
with col2:
  age = st.number_input(
      "Age (years)", min_value=5, max_value=100, value=25, step=1
  )

# Inputs: Measurements
st.header("Body Measurements")
col1, col2 = st.columns(2)
with col1:
  height = st.number_input(
      "Height (cm)", min_value=50.0, max_value=250.0, value=170.0, step=1.0
  )
with col2:
  weight = st.number_input(
      "Weight (kg)", min_value=10.0, max_value=300.0, value=60.0, step=1.0
  )

waist = st.number_input(
    "Waist Circumference (cm)", min_value=30.0, max_value=200.0, value=80.0, step=1.0
)
bmi = weight / ((height / 100) ** 2)
st.write(f"**Calculated BMI:** {bmi:.2f} kg/m²")

# Inputs: Lifestyle
st.header("Lifestyle Information")
activity = st.selectbox("Activity Level", ["Sedentary", "Moderately Active", "Active"])
alcohol = st.selectbox(
    "Alcohol Consumption", ["Non-drinker", "Light", "Moderate", "Heavy"]
)
smoking = st.selectbox("Smoking Status", ["Never", "Past", "Current"])

# Prediction Trigger
if st.button("Predict Risk", type="primary", use_container_width=True):
  # Map inputs to model encoding
  gender_val = 1 if gender == "Male" else 2
  activity_val = 0 if activity == "Sedentary" else 1
  alcohol_val = {"Non-drinker": 1, "Light": 2, "Moderate": 3, "Heavy": 4}[
      alcohol
  ]
  smoking_val = {"Never": 1, "Past": 2, "Current": 3}[smoking]

  # Format dataframe for prediction
  input_data = pd.DataFrame([{
      "Sex, 1:Men, 2:Women": gender_val,
      "Baseline Age, yrs": age,
      "Baseline BMI, kg/m2": bmi,
      "Baseline Waist circumference(WC), cm": waist,
      "Baseline Habit of exercise 0/1": activity_val,
      (
          "Baseline Alcohol consumption, non:1, light:2, moderate:3, heavy:4"
      ): alcohol_val,
      "Baseline Smoking status, 1:never, 2:past, 3:current": smoking_val,
  }])

  # Calculate risk probability
  probs = model.predict_proba(input_data)[0]
  risk_score = probs[list(model.classes_).index(1)]

  # Display result
  st.divider()
  st.subheader("Prediction Result")

  if risk_score < 0.10:
    st.success("## LOW RISK")
  elif risk_score < 0.25:
    st.warning("## MODERATE RISK")
  else:
    st.error("## HIGH RISK")

  st.caption(
      "Educational prediction only using Logistic Regression. Not a medical"
      " diagnosis or clinically validated result."
  )