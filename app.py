from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import librosa
import classify_audio

app = Flask(__name__)
app = Flask(__name__, static_folder="public", static_url_path="/public")
app.config['UPLOAD_FOLDER'] = 'uploads/'  # Thư mục lưu file ghi âm tạm thời
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
sample_rate = 22050

model = classify_audio.model
scaler = classify_audio.scaler
label_encoder = classify_audio.label_encoder


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
        audio, sr = librosa.load(file_path, sr=sample_rate)
        logging.info(f"Audio loaded, sample rate: {sr}, shape: {len(audio_data)}")

        result, confidence = classify_full_audio(model, audio, sr)
        os.remove(file_path)  # Xóa file sau khi xử lý

        return jsonify({"result": result, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_audio():
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
        os.remove(file_path)  # Xóa file sau khi xử lý
        return jsonify({"label": predicted_label, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)