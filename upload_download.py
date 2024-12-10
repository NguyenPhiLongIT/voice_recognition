import pyrebase

# Cấu hình Firebase Pyrebase
firebaseConfig = {
'apiKey': "AIzaSyD7LDJEuvgxr2tBbErHYI4pMaMQw97vLTA",
  'authDomain': "voice-recognization-b03dc.firebaseapp.com",
  'databaseURL': "https://voice-recognization-b03dc-default-rtdb.asia-southeast1.firebasedatabase.app",
  'projectId': "voice-recognization-b03dc",
  'storageBucket': "voice-recognization-b03dc.firebasestorage.app",
  'messagingSenderId': "76205606772",
  'appId': "1:76205606772:web:7aaa9d9d23336af88e8e21",
  'measurementId': "G-PPPSW2DZ7B"
}
LOCAL_UPLOAD_PATH = "model/voice_recognition_model.keras"
CLOUD_MODEL_PATH = "models/voice.keras"
LOCAL_DOWNLOAD_PATH = "downloaded_models"
MODEL_FILE = "downloaded_models/voice.keras"

firebase = pyrebase.initialize_app(firebaseConfig)
storage = firebase.storage()

import os
os.makedirs("downloaded_models", exist_ok=True)

# Upload file lên Cloud Storage
def upload_file_pyrebase(storage, local_path, cloud_path):
    try:
        storage.child(cloud_path).put(local_path)
        print(f"File uploaded to {cloud_path}")
    except Exception as e:
        print(f"Error uploading file: {e}")

# Download file từ Cloud Storage
def download_file_pyrebase():
    try:
        storage.child(CLOUD_MODEL_PATH).download(LOCAL_DOWNLOAD_PATH, MODEL_FILE)
        print(f"File downloaded to {LOCAL_DOWNLOAD_PATH}")
    except Exception as e:
        print(f"Error downloading file: {e}")

