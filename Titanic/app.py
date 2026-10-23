import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

st.title("🚢 Titanic Passenger Survival Predictor")
st.write("Enter the passenger details to get a prediction from the provided teaching model.")
st.caption("Educational demonstration only. Predictions are based on a small teaching dataset and are not historically reliable.")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "svc_trained_model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error("Could not load svc_trained_model.pkl. Make sure it is in the same folder as app.py.")
    st.stop()

with st.form("passenger_form"):
    pclass = st.selectbox("Passenger class", ["First", "Second", "Third"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    sibling = st.selectbox("Siblings/spouses aboard", ["Zero", "One", "Two", "Three"])
    embarked = st.selectbox("Port of embarkation", ["Cherbourg", "Queenstown", "Southampton"])
    submitted = st.form_submit_button("Predict survival")

# These mappings follow sklearn LabelEncoder's alphabetical class ordering,
# consistent with the sample encoded dataset supplied with the assignment.
pclass_map = {"First": 0, "Second": 1, "Third": 2}
gender_map = {"Female": 0, "Male": 1}
sibling_map = {"One": 0, "Three": 1, "Two": 2, "Zero": 3}
embarked_map = {"Cherbourg": 0, "Queenstown": 1, "Southampton": 2}

if submitted:
    inputs = pd.DataFrame([{
        "PClass": pclass_map[pclass],
        "Gender": gender_map[gender],
        "Sibling": sibling_map[sibling],
        "Embarked": embarked_map[embarked]
    }], columns=["PClass", "Gender", "Sibling", "Embarked"])

    try:
        prediction = model.predict(inputs)[0]
        if int(prediction) == 1:
            st.success("Prediction: Survived")
        else:
            st.warning("Prediction: Did not survive")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(inputs)[0]
            st.write("Model probability estimates:")
            st.write(f"Did not survive: {probabilities[0]:.1%}")
            st.write(f"Survived: {probabilities[1]:.1%}")
        st.write("Encoded input used:")
        st.dataframe(inputs, hide_index=True, use_container_width=True)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
