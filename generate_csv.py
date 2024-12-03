import os
import librosa
import numpy as np
import pandas as pd

# Hàm để trích xuất đặc trưng từ file .wav
def extract_features(file_name):
    # Tải file âm thanh
    audio, sample_rate = librosa.load(file_name, sr=16000)  # Sử dụng sample_rate=16000 như yêu cầu
    # Trích xuất các đặc trưng MFCC
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    # Các đặc trưng khác có thể bổ sung nếu cần (chroma, mel)
    return mfccs_mean

# Hàm để tạo file .csv chứa các đặc trưng cho mỗi file .wav
def process_directory(directory_path, output_csv):
    features_list = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                features = extract_features(file_path)
                features_list.append([file] + list(features))

    # Tạo DataFrame và lưu thành file CSV
    columns = ['filename'] + [f'mfcc_{i+1}' for i in range(len(features))]
    features_df = pd.DataFrame(features_list, columns=columns)
    features_df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")

# Đường dẫn đến thư mục chứa file .wav và đường dẫn file .csv đầu ra
directory_path = 'dataset'  # Đổi thành thư mục chứa file .wav của bạn
output_csv = 'audio_features.csv'  # Đổi thành đường dẫn lưu file .csv

# Gọi hàm để tạo file .csv chứa các đặc trưng
process_directory(directory_path, output_csv)
