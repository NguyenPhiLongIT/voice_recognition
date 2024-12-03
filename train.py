# Baseline
import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
import random

alert_audio_path = "dataset/alert"  
stop_audio_path = "dataset/stop" 
sample_rate = 22050  
segment_duration = 0.5 
overlap = 0.25 

def augment_audio_add_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    return augmented_audio

def load_audio_files(folder_path, label, sr=sample_rate, augment=False):
    data = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith('.wav'):
            audio, _ = librosa.load(file_path, sr=sr)
            
            if augment and random.random() < 0.5:
                audio = augment_audio_add_noise(audio, noise_factor=0.005)
            
            data.extend(segment_and_extract_features(audio, label, sr))
    return data

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
        chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=segment)
        
        feature_dict = {
            'mfcc_mean': np.mean(mfccs, axis=1),
            'mfcc_std': np.std(mfccs, axis=1),
            'chroma': np.mean(chroma, axis=1),
            'spectral_centroid': np.mean(spectral_centroid),
            'spectral_bandwidth': np.mean(spectral_bandwidth),
            'rolloff': np.mean(rolloff),
            'zero_crossing_rate': np.mean(zero_crossing_rate),
            'label': label
        }
        features.append(feature_dict)
    return features

alert_data = load_audio_files(alert_audio_path, label='Alert', augment=False)
stop_data = load_audio_files(stop_audio_path, label='Stop', augment=True)
all_data = alert_data + stop_data

df = pd.DataFrame(all_data)

X = pd.concat([
    pd.DataFrame(df['mfcc_mean'].tolist()),
    pd.DataFrame(df['mfcc_std'].tolist()),
    pd.DataFrame(df['chroma'].tolist()),
    df[['spectral_centroid', 'spectral_bandwidth', 'rolloff', 'zero_crossing_rate']]
], axis=1)
y = df['label']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = to_categorical(y)

# Ensure all column names are strings
X.columns = X.columns.astype(str)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(2),
    Dropout(0.2),
    Conv1D(64, 3, activation='relu'),
    MaxPooling1D(2),
    Dropout(0.2),
    Flatten(),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(y.shape[1], activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=4, callbacks=[early_stopping])


loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")