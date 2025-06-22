import os
import librosa
import numpy as np
import pandas as pd 

# Constants
SAVE_PATH = './Preprocessed_Emotion_Data/'
N_FFT = 1024
N_MELS = 128
HOP_LENGTH = 128
MAX_AUDIO_LENGTH = 10  # seconds
TARGET_SR = 22025

# Emotion mapping
EMOTION_TO_IDX = {
    'neutral': 0,
    'happy': 1,
    'sad': 2,
    'angry': 3,
    'disgust': 4,
    'fearful': 5
}

# Create folder if needed
os.makedirs(SAVE_PATH, exist_ok=True)
print("Saving preprocessed data at:", SAVE_PATH)

# Normalization
def per_sample_normalize(spec):
    mean = np.mean(spec, axis=1, keepdims=True)
    std = np.std(spec, axis=1, keepdims=True)
    return (spec - mean) / (std + 1e-6)

# Pad or trim
def pad_or_trim(audio, sr=TARGET_SR, target_length=MAX_AUDIO_LENGTH):
    target_samples = target_length * sr
    return np.pad(audio, (0, max(0, target_samples - len(audio))))[:target_samples]

# Convert audio to mel spectrogram
def audio_to_melspectrogram(audio, sr=TARGET_SR):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=N_FFT, 
                                         hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return per_sample_normalize(mel_db)

# Preprocess a single audio file
def preprocess_single_audio(file_path, emotion_label=None, save_name=None):
    audio, sr = librosa.load(file_path, sr=TARGET_SR)
    audio = pad_or_trim(audio, sr)
    mel_spec = audio_to_melspectrogram(audio, sr)
    mel_spec = np.expand_dims(mel_spec, axis=0)  # shape: (1, n_mels, time)

    # Save if needed
    if save_name:
        np.save(os.path.join(SAVE_PATH, f'{save_name}_X.npy'), mel_spec)
        if emotion_label:
            label = np.array([EMOTION_TO_IDX.get(emotion_label)])
            np.save(os.path.join(SAVE_PATH, f'{save_name}_y.npy'), label)
    
    if emotion_label:
        return mel_spec, EMOTION_TO_IDX.get(emotion_label)
    return mel_spec

# Preprocess a full dataset (DataFrame)
def preprocess_dataset(df, dataset_name):
    X, y = [], []
    for idx, row in df.iterrows():
        audio, sr = librosa.load(row['file_path'], sr=TARGET_SR)
        audio = pad_or_trim(audio)
        mel_spec = audio_to_melspectrogram(audio, sr)
        X.append(mel_spec)
        y.append(EMOTION_TO_IDX[row['emotion']])
        if idx % 100 == 0:
            print(f"{dataset_name} - Processed {idx} samples")

    X = np.array(X)
    y = np.array(y)
    print(f"{dataset_name} - Done. X shape: {X.shape}, y shape: {y.shape}")

    np.save(os.path.join(SAVE_PATH, f'X_{dataset_name}.npy'), X)
    np.save(os.path.join(SAVE_PATH, f'y_{dataset_name}.npy'), y)

    return X, y
