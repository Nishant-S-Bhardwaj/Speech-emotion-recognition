import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score
import tensorflow as tf

# Function to extract MFCC features from an audio file
def extract_mfcc(file_path, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

# Function to load dataset from a single folder by parsing filenames for labels
def load_dataset(data_folder):
    labels = []
    mfccs = []
    for file in os.listdir(data_folder):
        if file.endswith('.wav'):
            file_path = os.path.join(data_folder, file)
            label = file.split(' ')[0]
            mfcc = extract_mfcc(file_path)
            mfccs.append(mfcc)
            labels.append(label)
    return np.array(mfccs), np.array(labels)

# Example usage
data_folder = r"E:\coding\python files_ser\New folder"
mfccs, labels = load_dataset(data_folder)

# Encode the labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(mfccs, labels_categorical, test_size=0.2, random_state=42)

# Reshape the data for the CNN
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Build the CNN model
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Conv1D(128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Conv1D(256, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(len(np.unique(labels)), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=32)

# Function to make predictions on a new .wav file
def predict(model, file_path):
    mfcc = extract_mfcc(file_path)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]
    prediction = model.predict(mfcc)
    predicted_label = le.inverse_transform([np.argmax(prediction)])
    confidence_level = np.max(prediction)
    return predicted_label[0], confidence_level

# Example usage
wav_file = r"E:\coding\python files_ser\sad_fear_test_file.wav"  # Your .wav file path for prediction
predicted_label, confidence_level = predict(model, wav_file)
print(f'Predicted label: {predicted_label}')
print(f'Confidence level: {confidence_level:.2f}')

# Evaluate the model on the test set
y_pred_test = model.predict(X_test)
y_pred_test_labels = np.argmax(y_pred_test, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test_labels, y_pred_test_labels)
print(f'Accuracy: {accuracy:.2f}')

# Calculate precision
precision = precision_score(y_test_labels, y_pred_test_labels, average='weighted')
print(f'Precision: {precision:.2f}')

# Function to plot the training history
def plot_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Plot the training history
plot_history(history)

# Function to plot RMS value and pitch for an audio file
def plot_rms_pitch(file_path):
    y, sr = librosa.load(file_path, sr=None)
    rms = librosa.feature.rms(y=y)[0]
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    pitch = [pitches[t].max() if magnitudes[t].max() > 0.1 else 0 for t in range(pitches.shape[1])]
    
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(rms, label='RMS Value')
    plt.title('RMS Value')
    plt.xlabel('Frame')
    plt.ylabel('RMS')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(pitch, label='Pitch')
    plt.title('Pitch')
    plt.xlabel('Frame')
    plt.ylabel('Frequency (Hz)')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Plot RMS value and pitch for the prediction file
plot_rms_pitch(wav_file)
