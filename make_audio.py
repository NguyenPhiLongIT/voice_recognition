import os
import librosa
import numpy as np
import soundfile as sf

SAMPLE_RATE = 22050  
alert_path = "data/alert"
stop_path = "data/stop"


def add_noise(audio, noise_factor=0.005):
    noise = np.random.normal(0, 1, len(audio))  # Tạo nhiễu Gaussian
    return audio + noise_factor * noise

def pitch_shift(audio, sr, n_steps):
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def time_stretch(audio, rate):
    return librosa.effects.time_stretch(audio, rate=rate)

def increase_volume(audio, factor=2.0):
    audio = audio * factor
    audio = np.clip(audio, -1.0, 1.0)
    return audio

def make_noisy(path):
    file_counter =321
    for file_name in os.listdir(path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(path, file_name)
            y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            
            # Thêm nhiễu vào tín hiệu âm thanh
            # noisy_audio = add_noise(y, noise_factor=0.005)
            # noisy_audio = pitch_shift(y, sr=SAMPLE_RATE, n_steps=2)   
            # noisy_audio = time_stretch(y, rate=1.15)
            noisy_audio = time_stretch(y, rate=0.85)
            # noisy_audio = increase_volume(y, factor=2)
            augmented_file_name = f"stop{file_counter:02d}.wav"
            augmented_file_path = os.path.join(stop_path, augmented_file_name)

            # Lưu file đã xử lý
            sf.write(augmented_file_path, noisy_audio, sr)

            # Tăng biến đếm
            file_counter += 1
            # Lưu file mới
            # augmented_file_path = os.path.join(stop_louder, f"{file_name}")
            # sf.write(augmented_file_path, noisy_audio, sr)


make_noisy(stop_path)
