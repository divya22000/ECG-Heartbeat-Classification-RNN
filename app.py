import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from predict import load_trained_model, predict_heartbeat, CLASS_MAPPING, TARGET_TIMESTEPS

st.set_page_config(
    page_title="ECG Arrhythmia Classifier",
    page_icon="🫀",
    layout="wide"
)

# Cache model loading
@st.cache_resource
def get_model():
    try:
        return load_trained_model()
    except Exception as err:
        st.error(f"Error loading model: {err}")
        return None

model = get_model()

st.title("🫀 Universal ECG Classification Dashboard")
st.markdown(
    "Upload **any ECG CSV dataset** (MIT-BIH, PTB, or raw leads). "
    "The pipeline automatically handles variable sequence lengths, normalizes amplitudes, and executes RNN inference."
)

if model is None:
    st.stop()

# Sidebar: CSV File Upload
st.sidebar.header("Data Ingestion")
uploaded_file = st.sidebar.file_uploader(
    "Upload ECG CSV File",
    type=["csv"],
    help="Can contain single or multiple rows. Headers are optional."
)

selected_signal = None
actual_label = None

if uploaded_file is not None:
    try:
        # Detect if header is present
        sample_df = pd.read_csv(uploaded_file, nrows=2)
        has_header = not all(col.replace('.', '', 1).replace('-', '', 1).isdigit() for col in sample_df.columns)
        
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, header=0 if has_header else None)

        # Drop non-numeric columns (like IDs, timestamps, filenames)
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            st.error("No numeric ECG signal columns found in the uploaded file.")
            st.stop()

        num_rows, num_cols = numeric_df.shape
        st.sidebar.success(f"Loaded: {num_rows} heartbeat sample(s), {num_cols} features each.")

        # Let user select which heartbeat row to analyze
        if num_rows > 1:
            row_idx = st.sidebar.slider("Select Heartbeat Row to Analyze:", 0, num_rows - 1, 0)
        else:
            row_idx = 0

        raw_row = numeric_df.iloc[row_idx].values

        # If standard MIT-BIH dataset format (188 columns where last is the class index 0-4)
        if num_cols == 188:
            selected_signal = raw_row[:-1]
            actual_label = int(raw_row[-1])
        else:
            selected_signal = raw_row

    except Exception as e:
        st.error(f"Failed to process CSV file: {e}")
else:
    st.sidebar.info("Upload a CSV file to begin. Generating a demo waveform in the meantime.")
    # Fallback demo signal
    t = np.linspace(0, 1, TARGET_TIMESTEPS)
    selected_signal = (
        0.1 * np.exp(-((t - 0.2) ** 2) / 0.002)
        - 0.15 * np.exp(-((t - 0.35) ** 2) / 0.0005)
        + 1.0 * np.exp(-((t - 0.38) ** 2) / 0.0008)
        - 0.25 * np.exp(-((t - 0.41) ** 2) / 0.0005)
        + 0.25 * np.exp(-((t - 0.6) ** 2) / 0.006)
    )

# ---------------------------------------------------------
# Visualization & Predictions
# ---------------------------------------------------------
if selected_signal is not None:
    # Run prediction with auto-preprocessing (resampling + normalization)
    result = predict_heartbeat(model, selected_signal)
    clean_wave = result["preprocessed_signal"]

    col_wave, col_diag = st.columns([3, 2])

    with col_wave:
        st.subheader("Normalized Heartbeat Trace (187 Samples)")
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.plot(clean_wave, color="#0b63b6", lw=2, label="Processed Lead Signal")
        ax.set_xlim(0, TARGET_TIMESTEPS - 1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Timestep (Normalized)")
        ax.set_ylabel("Amplitude [0, 1]")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper right")
        st.pyplot(fig)
        plt.close(fig)

    with col_diag:
        st.subheader("Prediction Result")
        
        # Display classification banner
        if result["class_index"] == 0:
            st.success(f"### {result['label']}")
        else:
            st.error(f"### {result['label']}")

        st.metric(
            label="Confidence",
            value=f"{result['confidence'] * 100:.2f}%"
        )
        
        if actual_label is not None:
            actual_text = CLASS_MAPPING.get(actual_label, f"Class {actual_label}")
            match = (actual_label == result["class_index"])
            st.info(f"**Ground Truth Label in CSV:** {actual_text} {'✅ Match' if match else '⚠️ Mismatch'}")

        st.caption(result["description"])

    st.markdown("---")
    
    # Class probability bar chart
    st.subheader("Probability Distribution Across Arrhythmia Classes")
    prob_df = pd.DataFrame({
        "Class": [CLASS_MAPPING[i] for i in range(len(result["probabilities"]))],
        "Confidence (%)": [round(p * 100, 2) for p in result["probabilities"]]
    })
    st.bar_chart(prob_df.set_index("Class"))