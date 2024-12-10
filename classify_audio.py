from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np
import librosa
import pickle
import joblib

sample_rate = 22050
model_path = "model/voice_recognition_model.h5"
model = load_model(model_path)
segment_duration = 0.3
overlap = 0.15
scaler = joblib.load('model/scaler.pkl')
label_encoder = joblib.load('model/labelEncoder.pkl')


def segment_and_extract_features(audio, label, sr):
    features = []
    segment_samples = int(segment_duration * sr)
    overlap_samples = int(overlap * sr)
    num_segments = (len(audio) - segment_samples) // (segment_samples - overlap_samples) + 1

    for i in range(num_segments):
        start = i * (segment_samples - overlap_samples)
        end = start + segment_samples
        segment = audio[start:end]

        mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
        delta_mfcc = librosa.feature.delta(mfccs)
        delta2_mfcc = librosa.feature.delta(mfccs, order=2)
        chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
        mel_spectrogram = librosa.feature.melspectrogram(y=segment, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=segment)
        rms = librosa.feature.rms(y=segment)
        spectral_flatness = librosa.feature.spectral_flatness(y=segment)

        feature_dict = {
            'mfcc_mean': np.mean(mfccs, axis=1),
            'mfcc_std': np.std(mfccs, axis=1),
            'delta_mfcc_mean': np.mean(delta_mfcc, axis=1),
            'delta_mfcc_std': np.std(delta_mfcc, axis=1),
            'delta2_mfcc_mean': np.mean(delta2_mfcc, axis=1),
            'delta2_mfcc_std': np.std(delta2_mfcc, axis=1),
            'chroma_mean': np.mean(chroma, axis=1),
            'chroma_std': np.std(chroma, axis=1),
            'mel_spectrogram_mean': np.mean(mel_spectrogram, axis=1),
            'mel_spectrogram_std': np.std(mel_spectrogram, axis=1),
            'spectral_centroid': np.mean(spectral_centroid),
            'spectral_bandwidth': np.mean(spectral_bandwidth),
            'rolloff': np.mean(rolloff),
            'zero_crossing_rate': np.mean(zero_crossing_rate),
            'rms': np.mean(rms),
            'spectral_flatness': np.mean(spectral_flatness),
            'label': label
        }
        features.append(feature_dict)
    return features

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

def main_loop():
    print("Voice recognition process started. Press Ctrl+C to stop.")
    try:
        while True:
            # Ví dụ lấy audio từ tệp (có thể thay bằng stream hoặc input thực tế)
            audio_path = input("Nhập đường dẫn audio cần xử lý (hoặc 'exit' để thoát): ").strip()
            if audio_path.lower() == 'exit':
                print("Kết thúc chương trình.")
                break

            try:
                audio, sr = librosa.load(audio_path, sr=sample_rate)
                predicted_label, confidence = classify_full_audio(model, audio, sr)

                print(f"Dự đoán: {predicted_label}, Độ tin cậy: {confidence:.2f}")
            except Exception as e:
                print(f"Lỗi khi xử lý tệp: {e}")
    except KeyboardInterrupt:
        print("\nĐã dừng tiến trình.")

# file_path = "dataset/stop/stop02.wav"
# audio, _ = librosa.load(file_path, sr=sample_rate)
# full_audio_classification = classify_full_audio(model, audio)
# print(f"Full audio classification: {full_audio_classification}")

if __name__ == "__main__":
    main_loop()