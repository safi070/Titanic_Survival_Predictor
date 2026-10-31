import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Light custom styling for a cleaner look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Tighten vertical spacing a bit */
    .block-container {padding-top: 1.8rem; padding-bottom: 2rem;}
    
    /* Make the form look more card-like */
    div[data-testid="stForm"] {
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.4rem 1.2rem 1rem 1.2rem;
        background-color: #fafafa;
    }
    
    /* Result boxes */
    .result-survived {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 6px solid #28a745;
        padding: 1.2rem 1.4rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .result-not-survived {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left: 6px solid #ffc107;
        padding: 1.2rem 1.4rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        border: 1px solid #e9ecef;
    }
    </style>
    """,
    unsafe_allow_html=True,
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
# Header
# ---------------------------------------------------------------------------
st.title("🚢 Titanic Passenger Survival Predictor")
st.markdown(
    "Enter passenger details below to get a **teaching-model** prediction."
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
    st.subheader("Passenger details")

    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox(
            "Passenger class",
            options=list(PCLASS_MAP.keys()),
            help="Ticket class (1st, 2nd or 3rd)",
        )
        gender = st.selectbox(
            "Gender",
            options=list(GENDER_MAP.keys()),
        )

    with col2:
        sibling = st.selectbox(
            "Siblings / spouses aboard",
            options=list(SIBLING_MAP.keys()),
            help="Number of siblings or spouses travelling with the passenger",
        )
        embarked = st.selectbox(
            "Port of embarkation",
            options=list(EMBARKED_MAP.keys()),
            help="Where the passenger boarded the ship",
        )

    submitted = st.form_submit_button(
        "Predict survival",
        type="primary",
        use_container_width=True,
    )

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

        # ---- Main result card ----
        if prediction == 1:
            st.markdown(
                """
                <div class="result-survived">
                    <h3 style="margin:0 0 0.3rem 0;">🎉 Prediction: Survived</h3>
                    <p style="margin:0; opacity:0.85;">The model predicts this passenger would have survived.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="result-not-survived">
                    <h3 style="margin:0 0 0.3rem 0;">Prediction: Did not survive</h3>
                    <p style="margin:0; opacity:0.85;">The model predicts this passenger would not have survived.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ---- Probability estimates (safe check) ----
        # sklearn's SVC still has the method when probability=False,
        # so we must catch the actual error or check the flag carefully.
        show_proba = False
        proba = None

        if getattr(model, "probability", False):
            try:
                proba = model.predict_proba(input_data)[0]
                show_proba = True
            except Exception:
                # Fallback if something unexpected happens
                show_proba = False

        if show_proba and proba is not None:
            st.subheader("Model probability estimates")
            c1, c2 = st.columns(2)
            c1.metric("Did not survive", f"{proba[0]:.1%}")
            c2.metric("Survived", f"{proba[1]:.1%}")
        else:
            st.info(
                "Probability estimates are unavailable for this trained model "
                "(it was fitted with `probability=False`). "
                "The survival prediction above is still valid."
            )

        # ---- Debug / transparency ----
        with st.expander("Show encoded input used by the model"):
            st.dataframe(input_data, hide_index=True, use_container_width=True)
            st.caption(
                "These are the numeric values the model actually received "
                "(after LabelEncoder-style mapping)."
            )

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
