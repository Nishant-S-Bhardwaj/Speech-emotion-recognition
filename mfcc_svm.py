"""
Speech Emotion Recognition using MFCC features and SVM classifier.

This script extracts MFCC, RMS, and pitch features from audio files, trains an SVM classifier, and evaluates its performance.
"""

import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import logging
from typing import Tuple, Optional, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_features(file_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        rms = librosa.feature.rms(y=audio)
        rms_scaled = np.mean(rms.T, axis=0)
        pitches, magnitudes = librosa.core.piptrack(y=audio, sr=sample_rate)
        pitch = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch.append(pitches[index, t])
        pitch = np.array(pitch)
        pitch = pitch[pitch > 0]
        pitch_scaled = np.mean(pitch) if pitch.size > 0 else 0.0
        combined_features = np.hstack([mfccs_scaled, rms_scaled, pitch_scaled])
        return combined_features, rms[0]
    except Exception as e:
        logging.error(f"Error encountered while parsing file: {file_name}: {e}")
        return None, None

audio_dataset_path = r"E:\coding\python files_ser\New folder"
features: List[np.ndarray] = []
labels: List[str] = []
rms_values: List[np.ndarray] = []

for file in os.listdir(audio_dataset_path):
    if file.endswith('.wav'):
        file_path = os.path.join(audio_dataset_path, file)
        if 'happy' in file:
            emotion = 'happy'
        elif 'sad' in file:
            emotion = 'sad'
        elif 'fear' in file:
            emotion = 'fear'
        elif 'disgust' in file:
            emotion = 'disgust'
        elif 'angry' in file:
            emotion = 'angry'
        else:
            emotion = 'neutral'
        feature, rms_value = extract_features(file_path)
        if feature is not None:
            features.append(feature)
            labels.append(emotion)
            rms_values.append(rms_value)

features = np.array(features)
labels = np.array(labels)

if len(features) == 0 or len(labels) == 0:
    raise ValueError("No valid audio files found in the dataset path.")

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logging.info(f'Accuracy: {accuracy * 100:.2f}%')

def predict_emotion(file_name: str) -> Tuple[str, float, Optional[np.ndarray]]:
    feature, rms_value = extract_features(file_name)
    if feature is None:
        return "Error processing file", 0.0, None
    feature = feature.reshape(1, -1)
    prediction = model.predict(feature)
    prediction_proba = model.predict_proba(feature)
    emotion = label_encoder.inverse_transform(prediction)[0]
    confidence = prediction_proba[0][prediction][0]
    return emotion, confidence, rms_value

new_file = r"E:\coding\python files_ser\sad_fear_test_file.wav"
emotion, confidence, rms_value = predict_emotion(new_file)
logging.info(f'Predicted emotion: {emotion} with confidence: {confidence:.2f}')

def plot_rms_values(rms_values: List[np.ndarray], labels: List[str], test_rms: Optional[np.ndarray] = None) -> None:
    fig, axes = plt.subplots(6, 1, figsize=(10, 24))
    emotions = ['happy', 'sad', 'fear', 'disgust', 'angry', 'neutral']
    colors = ['blue', 'red', 'purple', 'brown', 'orange', 'grey']
    for i, emotion in enumerate(emotions):
        emotion_rms = [rms for rms, label in zip(rms_values, labels) if label == emotion]
        for idx, rms in enumerate(emotion_rms):
            axes[i].plot(rms, label=f'{emotion.capitalize()} RMS' if idx == 0 else "", color=colors[i])
        axes[i].set_title(f'RMS Values for {emotion.capitalize()} Emotions')
        axes[i].set_xlabel('Frame')
        axes[i].set_ylabel('RMS Value')
        axes[i].legend()
        if test_rms is not None:
            axes[i].plot(test_rms, label='Test RMS', color='green')
    plt.tight_layout()
    plt.show()

plot_rms_values(rms_values, labels, test_rms=rms_value)

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

class_names = label_encoder.classes_
plot_confusion_matrix(y_test, y_pred, class_names)