import os
import numpy as np
import librosa

SAMPLE_RATE = 8000  
DURATION = 4      
MAX_LEN = SAMPLE_RATE * DURATION

def load_and_preprocess_audio(file_path, sr=SAMPLE_RATE, duration=DURATION):
    y, _ = librosa.load(file_path, sr=sr, duration=duration)
    if len(y) < MAX_LEN:
        y = np.pad(y, (0, MAX_LEN - len(y)))  
    else:
        y = y[:MAX_LEN]  
    return y

def process_folder(folder_path, label, sr=SAMPLE_RATE, duration=DURATION):
    data = []
    labels = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            audio_data = load_and_preprocess_audio(file_path, sr, duration)
            data.append(audio_data)
            labels.append(label)
    return np.array(data), np.array(labels)


alert_path = "dataset/alert"
stop_path = "dataset/stop"

alert_data, alert_labels = process_folder(alert_path, label=0)  
stop_data, stop_labels = process_folder(stop_path, label=1)  

#kết hợp dữ liệu và nhãn
X = np.concatenate((alert_data, stop_data), axis=0)
y = np.concatenate((alert_labels, stop_labels), axis=0)

#chia dữ liệu thành tập huấn luyện và kiểm tra (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = np.expand_dims(X_train, axis=2)  # Shape: (samples, timesteps, 1)
X_test = np.expand_dims(X_test, axis=2)
N_CLASSES=2
y_train = to_categorical(y_train, num_classes=N_CLASSES)
y_test = to_categorical(y_test, num_classes=N_CLASSES)


model = Sequential([
    Conv1D(8, kernel_size=3, activation='relu', input_shape=(MAX_LEN, 1)),  
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    
    Conv1D(16, kernel_size=3, activation='relu'), 
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    
    LSTM(16, return_sequences=True), 
    Flatten(),
    
    Dense(32, activation='relu'),  
    Dropout(0.5),
    Dense(N_CLASSES, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=2,
    callbacks=[early_stopping]
)


test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")