from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import librosa
import classify_audio
from playsound import playsound
import threading

app = Flask(__name__, static_folder="public", static_url_path="/public")
app.config['UPLOAD_FOLDER'] = 'uploads/'  # Thư mục lưu file ghi âm tạm thời
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = classify_audio.model
scaler = classify_audio.scaler
label_encoder = classify_audio.label_encoder

ALERT_SOUND_PATH = "./public/alert_sound.wav"

alert_playing = False

def play_alert_sound():
    global alert_playing
    while alert_playing:
        playsound(ALERT_SOUND_PATH)


@app.route('/')
def index():
    return render_template('index.html')

@app.route("/classify-realtime", methods=["POST"])
def classify_realtime():
    if "audio" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(audio_file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    audio_file.save(file_path)

    try:
        audio, sr = librosa.load(file_path, sr=classify_audio.sample_rate)
        os.remove(file_path)  # Xóa file sau khi xử lý
        logging.info(f"Audio loaded, sample rate: {sr}, shape: {len(audio_data)}")
        predicted_label, confidence = classify_audio.classify_full_audio(model, audio, sr)
        confidence = float(confidence)
        return jsonify({"label": predicted_label, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_audio():
    global alert_playing
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        # Phân loại âm thanh
        audio, sr = librosa.load(file_path, sr=classify_audio.sample_rate)
        predicted_label, confidence = classify_audio.classify_full_audio(model, audio, sr)
        confidence = float(confidence)
        os.remove(file_path)

        if predicted_label == "Alert":
            if not alert_playing:
                alert_playing = True
                threading.Thread(target=play_alert_sound, daemon=True).start()
        elif predicted_label == "Stop":
            alert_playing = False

        return jsonify({"label": predicted_label, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)