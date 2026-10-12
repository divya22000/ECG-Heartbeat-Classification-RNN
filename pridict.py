import os
import numpy as np
import tensorflow as tf
from scipy.interpolate import interp1d

CLASS_MAPPING = {
    0: "Normal Beat (N)",
    1: "Supraventricular Ectopic Beat (S)",
    2: "Ventricular Ectopic Beat (V)",
    3: "Fusion Beat (F)",
    4: "Unknown / Unclassifiable (Q)"
}

CLASS_DESCRIPTIONS = {
    0: "Standard rhythm without significant morphological deviations.",
    1: "Premature beat originating in the atria above the ventricles.",
    2: "Premature beat originating in the ventricular myocardium.",
    3: "Hybrid morphology resulting from simultaneous supraventricular and ventricular activation.",
    4: "Paced beat or unclassifiable artifact."
}

TARGET_TIMESTEPS = 187

def load_trained_model(model_path=os.path.join("model", "ecg_heartbeat_rnn.keras")):
    """Loads and caches the trained Keras RNN model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return tf.keras.models.load_model(model_path)

def preprocess_ecg_signal(signal_sample, target_len=TARGET_TIMESTEPS):
    """
    Cleans, resamples, and normalizes an arbitrary ECG signal sequence:
    1. Removes NaNs/Infs.
    2. Resamples/interpolates signal to exactly target_len points.
    3. Min-Max normalizes signal amplitude between [0, 1].
    """
    signal = np.array(signal_sample, dtype=np.float32).flatten()
    
    # Handle NaN or Inf values
    if np.isnan(signal).any() or np.isinf(signal).any():
        signal = np.nan_to_num(signal, nan=0.0, posinf=1.0, neginf=0.0)

    # 1. Resample to target_len points using linear interpolation
    current_len = len(signal)
    if current_len != target_len:
        x_old = np.linspace(0, 1, current_len)
        x_new = np.linspace(0, 1, target_len)
        f_interp = interp1d(x_old, signal, kind='linear', fill_value="extrapolate")
        signal = f_interp(x_new)

    # 2. Dynamic Min-Max Normalization to [0, 1] range expected by RNN
    s_min, s_max = np.min(signal), np.max(signal)
    if s_max - s_min > 1e-6:
        signal = (signal - s_min) / (s_max - s_min)
    else:
        signal = np.zeros_like(signal)

    return signal

def predict_heartbeat(model, signal_sample):
    """
    Preprocesses the raw signal and generates the RNN prediction.
    """
    clean_signal = preprocess_ecg_signal(signal_sample, target_len=TARGET_TIMESTEPS)
    
    # Reshape to expected input shape: (1, 187, 1)
    input_tensor = clean_signal.reshape(1, TARGET_TIMESTEPS, 1)

    probabilities = model.predict(input_tensor, verbose=0)[0]
    predicted_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_idx])
    
    return {
        "preprocessed_signal": clean_signal,
        "class_index": predicted_idx,
        "label": CLASS_MAPPING.get(predicted_idx, "Unknown"),
        "confidence": confidence,
        "probabilities": probabilities.tolist(),
        "description": CLASS_DESCRIPTIONS.get(predicted_idx, "")
    }