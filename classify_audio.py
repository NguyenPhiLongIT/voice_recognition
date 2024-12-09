from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import librosa

sample_rate = 22050
model_path = "voice_recognition_model.keras"
model = load_model(model_path)

def classify_full_audio(model, audio, sr=sample_rate):
    segments = segment_and_extract_features(audio, label=None, sr=sr)

    segment_features = pd.concat([
        pd.DataFrame([seg['mfcc_mean'] for seg in segments]),
        pd.DataFrame([seg['mfcc_std'] for seg in segments]),
        # pd.DataFrame([seg['chroma'] for seg in segments]),
        pd.DataFrame([seg['chroma_mean'] for seg in segments]),  
        pd.DataFrame([seg['chroma_std'] for seg in segments]),
        pd.DataFrame([[
            seg['spectral_centroid'],
            seg['spectral_bandwidth'],
            seg['rolloff'],
            seg['zero_crossing_rate']] for seg in segments]
        )
    ], axis=1)
    segment_features = scaler.transform(segment_features)
    segment_features = segment_features.reshape(segment_features.shape[0], segment_features.shape[1], 1)

    segment_predictions = model.predict(segment_features)
    segment_labels = np.argmax(segment_predictions, axis=1)

    confidence_scores = segment_predictions.max(axis=1)
    avg_confidence = np.mean(confidence_scores)
    confidence_threshold = 0.72
    if avg_confidence < confidence_threshold:
        return "Unknown", avg_confidence

    full_audio_prediction = np.argmax(np.bincount(segment_labels))
    predicted_label = label_encoder.inverse_transform([full_audio_prediction])[0]

    return predicted_label, avg_confidence

# Example
file_path = "dataset/alert/alert01.wav"
audio, _ = librosa.load(file_path, sr=sample_rate)
full_audio_classification = classify_full_audio(model, audio)
print(f"Full audio classification: {full_audio_classification}")