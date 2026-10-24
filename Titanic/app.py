import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Constants & mappings (created once)
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "svc_trained_model.pkl"

# These follow the LabelEncoder alphabetical ordering used when the model
# was trained.
PCLASS_MAP = {"First": 0, "Second": 1, "Third": 2}
GENDER_MAP = {"Female": 0, "Male": 1}
SIBLING_MAP = {"Zero": 3, "One": 0, "Two": 2, "Three": 1}
EMBARKED_MAP = {"Cherbourg": 0, "Queenstown": 1, "Southampton": 2}

FEATURE_COLUMNS = ["PClass", "Gender", "Sibling", "Embarked"]

# ---------------------------------------------------------------------------
# Cached model loader
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(path: Path):
    """Load the trained model once and keep it in memory."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {path}")
    return joblib.load(path)

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🚢 Titanic Passenger Survival Predictor")
st.write(
    "Enter the passenger details to get a prediction from the teaching model."
)
st.caption(
    "Educational demonstration only. Predictions are based on a small teaching "
    "dataset and are **not** historically reliable."
)

# Load model early so we can stop cleanly if it fails
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(
        f"Could not load the model (`{MODEL_PATH.name}`).\n\n"
        "Make sure the file is in the same folder as `app.py`."
    )
    st.exception(e)
    st.stop()

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
with st.form("passenger_form", clear_on_submit=False):
    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Passenger class", list(PCLASS_MAP.keys()))
        gender = st.selectbox("Gender", list(GENDER_MAP.keys()))

    with col2:
        sibling = st.selectbox("Siblings / spouses aboard", list(SIBLING_MAP.keys()))
        embarked = st.selectbox("Port of embarkation", list(EMBARKED_MAP.keys()))

    submitted = st.form_submit_button("Predict survival", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if submitted:
    # Build the exact feature vector the model expects
    input_data = pd.DataFrame(
        [
            {
                "PClass": PCLASS_MAP[pclass],
                "Gender": GENDER_MAP[gender],
                "Sibling": SIBLING_MAP[sibling],
                "Embarked": EMBARKED_MAP[embarked],
            }
        ],
        columns=FEATURE_COLUMNS,
    )

    try:
        prediction = int(model.predict(input_data)[0])

        # Main result
        if prediction == 1:
            st.success("### Prediction: **Survived** 🎉")
        else:
            st.warning("### Prediction: **Did not survive**")

        # Probability estimates (if available)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0]
            st.subheader("Model probability estimates")
            c1, c2 = st.columns(2)
            c1.metric("Did not survive", f"{proba[0]:.1%}")
            c2.metric("Survived", f"{proba[1]:.1%}")

        # Debug / transparency
        with st.expander("Show encoded input used by the model"):
            st.dataframe(input_data, hide_index=True, use_container_width=True)

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
