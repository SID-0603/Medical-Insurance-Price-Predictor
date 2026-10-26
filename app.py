import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
#import warnings

#warnings.filterwarnings("ignore")

st.set_page_config(
    page_title = "Insurance Cost Predictor",
    layout = "wide"
)


MODEL_PATH = "insurance_model.pkl"
FEATURE_PATH = "model_feature.pkl"

REF_AVG_NON_SMOKER = 8440
REF_AVG_SMOKER = 32050

@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURE_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURE_PATH)
    return model, feature_cols


def bmi_category(bmi):
    if bmi < 18.5:
        return "underweight"

    elif 18.5 <= bmi < 25:
        return "normal"

    elif 25 <= bmi < 30:
        return "overweight"

    else:
        return "obese"

def build_input_row(age, sex, bmi, children, smoker, region, feature_cols):
    category = bmi_category(bmi)
    
    row = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex_male": 1 if sex == "male" else 0,
        "smoker_yes": 1 if smoker == "yes" else 0,
        "region": region,
        "bmi_category": category,
    }
    encoded = pd.DataFrame([row])
    return encoded.reindex(columns=feature_cols, fill_value=0)

model, feature_cols = load_artifacts()

st.title("Medical Insurance Price Predictor")

if model is None:
    st.error("Failed to load the model.")
    st.stop()

left, right = st.columns([1, 1.2], gap = "large")

with left:
    st.subheader("Patient Details")

    age = st.slider("Age", 18, 64, 30)
    sex = st.radio("Sex", ["male", "female"], horizontal = True)
    bmi = st.number_input("BMI", 15.0, 55.0,26.0, 0.1)
    children = st.slider("Number of Children", 0, 5, 0)
    smoker = st.radio("Smoker", ["yes", "no"], horizontal = True)
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    predict_clicked = st.button("Estimate Charges", type = "primary", use_container_width = True)

with right:
    st.subheader("Estimate Charges")

    if predict_clicked:
        input_row = build_input_row(
            age, sex, bmi, children, smoker, region, feature_cols
        )

        log_pred = model.predict(input_row)[0]
        pred_charges = np.expm1(log_pred)

        st.metric(
            "Predicted annual charge",
            f"Rs. {pred_charges:,.2f}"
        )

        st.divider()

        compare_df = pd.DataFrame({
            "Group": [
                "Your prediction",
                "Non-smoker avg.",
                "Smoker avg."
            ],
            "Charges": [
                pred_charges,
                REF_AVG_NON_SMOKER,
                REF_AVG_SMOKER
            ]
        }).set_index("Group")

        st.bar_chart(compare_df)

    else:
        st.info("Fill the details and click **Estimate Charges**.")


# & "C:\Users\Siddharth Kanodia\anaconda3\python.exe" -m streamlit run "C:\Users\Siddharth Kanodia\OneDrive\Desktop\Minor Project 7th sem\app.py"