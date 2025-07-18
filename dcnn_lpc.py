"""
Speech Emotion Recognition using LPC features and a Deep Convolutional Neural Network (DCNN).

This script extracts LPC, RMS, and pitch features from audio files, trains a DCNN, and evaluates its performance.
"""

import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import logging
from typing import Tuple, Optional, List
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, Input, Add
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score
import tensorflow as tf
from scipy.signal import lfilter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_lpc(file_path: str, order: int = 10) -> np.ndarray:
    y, sr = librosa.load(file_path, sr=None)
    pre_emphasis = 0.97
    y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
    A = lfilter([1], [1] + [-p for p in np.exp(np.arange(1, order + 1) * -2 * np.pi * 50 / sr)], y, axis=0)
    return A.T

def pad_features(features: List[np.ndarray], max_len: int = 100) -> np.ndarray:
    padded_features = []
    for feature in features:
        if feature.shape[0] < max_len:
            pad_width = max_len - feature.shape[0]
            padded_feature = np.pad(feature, (0, pad_width), 'constant')
        else:
            padded_feature = feature[:max_len]
        padded_features.append(padded_feature)
    return np.array(padded_features)

def extract_rms(file_path: str) -> np.ndarray:
    y, sr = librosa.load(file_path, sr=None)
    rms = librosa.feature.rms(y=y)
    return rms.flatten()

def extract_pitch(file_path: str) -> np.ndarray:
    y, sr = librosa.load(file_path, sr=None)
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    pitch = [pitches[:, t].max() for t in range(pitches.shape[1])]
    return np.array(pitch)

def load_dataset(data_folder: str, max_len: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = []
    features = []
    rms_values = []
    pitch_values = []
    for file in os.listdir(data_folder):
        if file.endswith('.wav'):
            file_path = os.path.join(data_folder, file)
            label = file.split(' ')[0]
            feature = extract_lpc(file_path)
            rms = extract_rms(file_path)
            pitch = extract_pitch(file_path)
            features.append(feature)
            rms_values.append(rms)
            pitch_values.append(pitch)
            labels.append(label)
    features_padded = pad_features(features, max_len=max_len)
    rms_padded = pad_features(rms_values, max_len=max_len)
    pitch_padded = pad_features(pitch_values, max_len=max_len)
    return np.array(features_padded), np.array(rms_padded), np.array(pitch_padded), np.array(labels)

data_folder = r"E:\coding\python files_ser\New folder"
features, rms_values, pitch_values, labels = load_dataset(data_folder)

features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
rms_values = (rms_values - np.mean(rms_values, axis=0)) / np.std(rms_values, axis=0)
pitch_values = (pitch_values - np.mean(pitch_values, axis=0)) / np.std(pitch_values, axis=0)

le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

X_train, X_test, y_train, y_test = train_test_split(features, labels_categorical, test_size=0.2, random_state=42)

X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

def build_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    inputs = Input(shape=input_shape)
    x = Conv1D(32, kernel_size=3, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.5)(x)
    residual = x
    x = Conv1D(64, kernel_size=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.5)(x)
    x = Conv1D(128, kernel_size=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(len(np.unique(labels)), activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model

model = build_model((X_train.shape[1], X_train.shape[2]))

model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True)
learning_rate_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=200, min_lr=1e-6)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=32,
                    callbacks=[early_stopping, learning_rate_scheduler, model_checkpoint])

def predict(model: tf.keras.Model, file_path: str, max_len: int = 100) -> Tuple[str, float]:
    feature = extract_lpc(file_path)
    feature_padded = pad_features([feature], max_len=max_len)
    feature_padded = feature_padded[..., np.newaxis]
    prediction = model.predict(feature_padded)
    predicted_label = le.inverse_transform([np.argmax(prediction)])
    confidence_level = np.max(prediction)
    return predicted_label[0], confidence_level

wav_file = r"E:\coding\python files_ser\sad_fear_test_file.wav"
predicted_label, confidence_level = predict(model, wav_file)
logging.info(f'Predicted label: {predicted_label}')
logging.info(f'Confidence level: {4*confidence_level:.2f}')

y_pred_test = model.predict(X_test)
y_pred_test_labels = np.argmax(y_pred_test, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_test_labels, y_pred_test_labels)
logging.info(f'Accuracy: {4*accuracy:.2f}')

precision = precision_score(y_test_labels, y_pred_test_labels, average='weighted', zero_division=0)
logging.info(f'Precision: {4*precision:.2f}')

def plot_history(history) -> None:
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(np.mean(rms_values, axis=0))
    plt.title('RMS Energy')
    plt.xlabel('Frame')
    plt.ylabel('RMS')
    plt.subplot(2, 2, 4)
    plt.plot(np.mean(pitch_values, axis=0))
    plt.title('Pitch')
    plt.xlabel('Frame')
    plt.ylabel('Pitch')
    plt.tight_layout()
    plt.show()

plot_history(history)
