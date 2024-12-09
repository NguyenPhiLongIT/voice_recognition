const recordBtn = document.getElementById("recordBtn");
const uploadInput = document.getElementById("uploadInput");
const uploadBtn = document.getElementById("uploadBtn");
const resultDiv = document.getElementById("result");

let recorder, audioBlob;

recordBtn.addEventListener("click", async () => {
    if (recordBtn.textContent === "Record") {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        recorder = new MediaRecorder(stream);
        recorder.start();
        console.log("Recording started");
        recorder.ondataavailable = (e) => {
            audioBlob = e.data;
        };

        recordBtn.textContent = "Stop";
    } else {
        recorder.stop();
        console.log("Recording stopped");
        recordBtn.textContent = "Record";
    }
});

uploadBtn.addEventListener("click", async () => {
    const formData = new FormData();

    if (audioBlob) {
        formData.append("audio", audioBlob, "recording.wav");
    } else if (uploadInput.files.length > 0) {
        formData.append("audio", uploadInput.files[0]);
    } else {
        resultDiv.textContent = "No audio selected!";
        return;
    }

    resultDiv.textContent = "Uploading...";

    const response = await fetch("/upload", {
        method: "POST",
        body: formData,
    });

    const result = await response.json();
    if (response.ok) {
        resultDiv.textContent = `Prediction: ${result.label}, Confidence: ${result.confidence.toFixed(2)}`;
    } else {
        resultDiv.textContent = `Error: ${result.error}`;
    }
});
