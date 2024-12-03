def classify_full_audio(model, audio, sr=sample_rate):
    segments = segment_and_extract_features(audio, label=None, sr=sr)
    
    segment_features = pd.concat([
        pd.DataFrame([seg['mfcc_mean'] for seg in segments]),
        pd.DataFrame([seg['mfcc_std'] for seg in segments]),
        pd.DataFrame([seg['chroma'] for seg in segments]),
        pd.DataFrame([[
            seg['spectral_centroid'], 
            seg['spectral_bandwidth'], 
            seg['rolloff'], 
            seg['zero_crossing_rate']] for seg in segments]
        )
    ], axis=1)
    segment_features = scaler.transform(segment_features)
    segment_features = segment_features.reshape(segment_features.shape[0], segment_features.shape[1], 1)
    
    # Predictions for each segment
    segment_predictions = model.predict(segment_features)
    segment_labels = np.argmax(segment_predictions, axis=1)  # Get class indices (0 for Real, 1 for Fake)
    
    # Classify the full audio
    full_audio_prediction = np.argmax(np.bincount(segment_labels))
    return label_encoder.inverse_transform([full_audio_prediction])[0]

# Example
file_path = "dataset/alert/alert01.wav"
audio, _ = librosa.load(file_path, sr=sample_rate)
full_audio_classification = classify_full_audio(model, audio)
print(f"Full audio classification: {full_audio_classification}")