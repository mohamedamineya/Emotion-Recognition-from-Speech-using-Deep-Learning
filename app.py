from flask import Flask, render_template, request
import os
import librosa
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict
import cv2
import uuid  # Pour noms uniques

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Constants
N_FFT = 1024
N_MELS = 128
HOP_LENGTH = 128
MAX_AUDIO_LENGTH = 10  # seconds
TARGET_SR = 22025
MODEL_PATH = 'model_finetuned_on_emodb.pth'

# Emotion labels
EMOTION_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Define model
class EmotionModel(nn.Module):
    def __init__(self, num_classes=8):
        super(EmotionModel, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EmotionModel()
checkpoint = torch.load(MODEL_PATH, map_location=device)

# Handle nested checkpoints
if isinstance(checkpoint, dict) and 'model' in checkpoint:
    checkpoint = checkpoint['model']

new_state_dict = OrderedDict()
for k, v in checkpoint.items():
    new_key = k.replace("model.", "") if k.startswith("model.") else k
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict, strict=False)
model.eval().to(device)

# Preprocessing
def per_sample_normalize(spec):
    mean = np.mean(spec, axis=1, keepdims=True)
    std = np.std(spec, axis=1, keepdims=True)
    return (spec - mean) / (std + 1e-6)

def pad_or_trim(audio, sr=TARGET_SR, target_length=MAX_AUDIO_LENGTH):
    target_samples = target_length * sr
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))
    else:
        audio = audio[:target_samples]
    return audio

def audio_to_melspectrogram(audio, sr=TARGET_SR):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = per_sample_normalize(mel_db)
    return mel_db

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'vocal-file' not in request.files:
        return render_template('result.html', error='No file uploaded')

    file = request.files['vocal-file']
    if file.filename == '':
        return render_template('result.html', error='Empty filename')

    # Clear previous uploads
    for f in os.listdir(UPLOAD_FOLDER):
        try:
            os.remove(os.path.join(UPLOAD_FOLDER, f))
        except:
            pass

    # Generate unique filename
    filename = str(uuid.uuid4()) + "_" + file.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    print(f"File uploaded: {filename}")  # Debugging log

    try:
        audio, sr = librosa.load(file_path, sr=TARGET_SR)
        print(f"Audio loaded: {file_path}, shape: {audio.shape}")

        audio = pad_or_trim(audio)
        mel_spec = audio_to_melspectrogram(audio, sr)
        print(f"Mel mean: {np.mean(mel_spec)}, shape: {mel_spec.shape}")

        mel_spec_resized = cv2.resize(mel_spec, (128, 128), interpolation=cv2.INTER_LINEAR)
        input_tensor = torch.tensor(mel_spec_resized).unsqueeze(0).unsqueeze(0).float().to(device)

        print(f"Input tensor shape: {input_tensor.shape}")

        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()
            emotion = EMOTION_LABELS[prediction]

        print(f"Predicted emotion: {emotion}")
        return render_template('result.html', emotion=emotion)

    except Exception as e:
        print(f"Error: {str(e)}")
        return render_template('result.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True)
