import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin, clone


# Required when loading the saved SPECT model.
# The class name and implementation must match the training script.
class VoteSelector(BaseEstimator, TransformerMixin):
    """Union selector: retain a feature if at least one selector selects it."""

    def __init__(self, selectors=None, min_votes=1):
        self.selectors = selectors
        self.min_votes = min_votes

    def fit(self, X, y):
        votes = np.zeros(X.shape[1], dtype=int)
        for selector in self.selectors:
            fitted = clone(selector).fit(X, y)
            votes += fitted.get_support().astype(int)
        self.support_ = votes >= self.min_votes
        if not self.support_.any():
            self.support_[:] = True
        return self

    def transform(self, X):
        return X[:, self.support_]

    def get_support(self):
        return self.support_


# -------------------- Page setup --------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT 
METADATA_DIR = ROOT 


@st.cache_data
def load_metadata():
    metadata = {}
    for json_path in sorted(METADATA_DIR.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as file:
            item = json.load(file)
        metadata[item["id"]] = item
    return metadata


@st.cache_resource
def load_model(model_path):
    # VoteSelector above must remain defined at module level for the SPECT pickle.
    return joblib.load(model_path)


def pretty_feature_name(feature):
    """Turn technical feature names into readable labels."""
    labels = {
        "cp": "Chest pain type (cp)",
        "trestbps": "Resting blood pressure (trestbps)",
        "chol": "Serum cholesterol (chol)",
        "fbs": "Fasting blood sugar (fbs)",
        "restecg": "Resting ECG result (restecg)",
        "thalach": "Maximum heart rate (thalach)",
        "exang": "Exercise-induced angina (exang)",
        "oldpeak": "ST depression (oldpeak)",
        "slope": "ST segment slope (slope)",
        "ca": "Major vessels (ca)",
        "thal": "Thalassemia (thal)",
        "anaemia": "Anaemia",
        "creatinine_phosphokinase": "Creatinine phosphokinase",
        "diabetes": "Diabetes",
        "ejection_fraction": "Ejection fraction",
        "high_blood_pressure": "High blood pressure",
        "platelets": "Platelets",
        "serum_creatinine": "Serum creatinine",
        "serum_sodium": "Serum sodium",
        "smoking": "Smoking",
        "time": "Follow-up period (time)",
        "sex": "Sex",
        "age": "Age",
    }
    return labels.get(feature, feature.replace("_", " ").title())


def input_step(feature, minimum, maximum):
    if feature in {
        "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal",
        "anaemia", "diabetes", "high_blood_pressure", "smoking",
    } or (minimum.is_integer() and maximum.is_integer()):
        return 1.0
    return 0.1


st.title("❤️ Heart Disease Prediction System")
st.caption(
    "Multi-dataset machine-learning demonstration using the trained models "
    "and dataset-specific input metadata."
)

try:
    all_metadata = load_metadata()
except Exception as exc:
    st.error(f"Could not read metadata files: {exc}")
    st.stop()

if not all_metadata:
    st.error("No JSON metadata files were found in the model_metadata folder.")
    st.stop()

dataset_ids = list(all_metadata.keys())
dataset_names = [all_metadata[key]["dataset"] for key in dataset_ids]
selected_name = st.sidebar.selectbox("Select dataset", dataset_names)
selected_id = dataset_ids[dataset_names.index(selected_name)]
meta = all_metadata[selected_id]

st.sidebar.markdown("---")
st.sidebar.subheader("Selected model")
st.sidebar.write(f"**Feature technique:** {meta['feature_technique']}")
st.sidebar.write(f"**Boosting algorithm:** {meta['booster']}")
st.sidebar.write(f"**Training records:** {meta['n_training_records']}")

st.subheader(f"{selected_name} — Patient information")
st.write(
    "Enter numeric values for the features below. The default values are "
    "the training-set medians recorded in this dataset's metadata."
)

features = meta["input_features"]
input_values = {}

# Use two columns for the input form. The feature order remains the metadata order.
with st.form("prediction_form"):
    columns = st.columns(2)
    for index, feature in enumerate(features):
        minimum = float(meta["minimums"][feature])
        maximum = float(meta["maximums"][feature])
        default = float(meta["medians"][feature])
        default = min(max(default, minimum), maximum)

        # Ensure Streamlit receives a valid step and range.
        step = input_step(feature, minimum, maximum)
        with columns[index % 2]:
            input_values[feature] = st.number_input(
                pretty_feature_name(feature),
                min_value=minimum,
                max_value=maximum,
                value=default,
                step=step,
                key=f"{selected_id}_{feature}",
                help=f"Training-data range: {minimum:g} to {maximum:g}",
            )

    submitted = st.form_submit_button(
        "Predict heart disease",
        type="primary",
        use_container_width=True,
    )

if submitted:
    model_path = MODEL_DIR / f"{selected_id}.joblib"

    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        st.stop()

    try:
        model = load_model(str(model_path))
        input_frame = pd.DataFrame(
            [[input_values[feature] for feature in features]],
            columns=features,
        )

        prediction = int(model.predict(input_frame)[0])

        st.markdown("---")
        st.subheader("Prediction result")

        label = meta.get("labels", {}).get(
            str(prediction),
            "Heart disease PRESENT" if prediction == 1 else "Heart disease ABSENT",
        )

        if prediction == 1:
            st.error(f"Model prediction: {label}")
        else:
            st.success(f"Model prediction: {label}")

        # Show probability only when the estimator supports it.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_frame)[0]
            classes = list(model.classes_)
            probability_by_class = {
                str(cls): float(prob)
                for cls, prob in zip(classes, probabilities)
            }

            absent_probability = probability_by_class.get("0")
            present_probability = probability_by_class.get("1")

            if absent_probability is not None and present_probability is not None:
                left, right = st.columns(2)
                left.metric("Model score — absent", f"{absent_probability * 100:.1f}%")
                right.metric("Model score — present", f"{present_probability * 100:.1f}%")
                st.caption(
                    "These are model output probabilities, not calibrated clinical "
                    "risk estimates."
                )

        with st.expander("View submitted feature values"):
            st.dataframe(input_frame, use_container_width=True)

    except Exception as exc:
        st.error(f"Prediction could not be completed: {exc}")
        st.info(
            "Check that the model files, metadata files, and Python packages "
            "are available in this project environment."
        )

st.markdown("---")
st.subheader("Dataset information")
info1, info2, info3 = st.columns(3)
info1.metric("Input features", len(features))
info2.metric("Training records", meta["n_training_records"])
info3.metric("Feature method", meta["feature_technique"])

class0 = int(meta.get("class_0_count", 0))
class1 = int(meta.get("class_1_count", 0))
if class0 + class1 > 0:
    st.write("Training dataset class distribution")
    distribution = pd.DataFrame(
        {"Records": [class0, class1]},
        index=["Heart disease absent (0)", "Heart disease present (1)"],
    )
    st.bar_chart(distribution)

st.warning(
    "Research and educational use only. This application is not a medical "
    "device and its predictions must not be used for diagnosis or treatment. "
    "Consult a qualified healthcare professional for medical decisions."
)
