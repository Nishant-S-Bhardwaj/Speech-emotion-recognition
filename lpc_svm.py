import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer

# Function to extract features from an audio file
def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0) if mfccs.size > 0 else np.zeros(40)
        
        # Extract RMS
        rms = librosa.feature.rms(y=audio)
        rms_scaled = np.mean(rms.T, axis=0) if rms.size > 0 else 0.0
        
        # Extract Pitch
        pitches, magnitudes = librosa.core.piptrack(y=audio, sr=sample_rate)
        pitch = [pitches[index, t] for t in range(pitches.shape[1]) if (index := magnitudes[:, t].argmax()) > 0]
        pitch_scaled = np.mean(pitch) if len(pitch) > 0 else 0.0
        
        # Extract LPC
        lpc_coeffs = librosa.lpc(audio, order=13) if audio.size > 0 else np.zeros(13)
        
        # Combine selected features
        combined_features = np.hstack([mfccs_scaled, rms_scaled, pitch_scaled, lpc_coeffs])
        
        return combined_features, rms[0]
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}")
        return None, None

# Directory where your audio files are stored
audio_dataset_path = r"E:\\coding\\python files_ser\\New folder"

# List to store extracted features and labels
features = []
labels = []
rms_values = []

# Load the dataset
emotion_classes = ['happy', 'sad', 'neutral', 'angry', 'fear', 'disgust']

for file in os.listdir(audio_dataset_path):
    if file.endswith('.wav'):
        file_path = os.path.join(audio_dataset_path, file)
        emotion = next((emotion for emotion in emotion_classes if emotion in file), None)
        if emotion is not None:
            feature, rms_value = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(emotion)
                rms_values.append(rms_value)

# Convert features and labels to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Ensure we have enough data
if len(features) == 0 or len(labels) == 0:
    raise ValueError("No valid audio files found in the dataset path.")

# Impute missing values
imputer = SimpleImputer(strategy='mean')
features = imputer.fit_transform(features)

# Encode the labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

# Train a Support Vector Machine (SVM) classifier
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Function to predict the emotion of a given audio file
def predict_emotion(file_name):
    feature, rms_value = extract_features(file_name)
    if feature is None:
        return "Error processing file", 0.0, None
    feature = feature.reshape(1, -1)
    feature = imputer.transform(feature)  # Apply imputation to new data
    prediction = model.predict(feature)
    prediction_proba = model.predict_proba(feature)
    emotion = label_encoder.inverse_transform(prediction)[0]
    confidence = prediction_proba[0][prediction][0]
    return emotion, confidence, rms_value

# Specify the file to test
test_file = r"E:\coding\python files_ser\sad_fear_test_file.wav"
emotion, confidence, rms_value = predict_emotion(test_file)
print(f'Predicted emotion: {emotion} with confidence: {confidence:.2f}')

# Visualizing the RMS values for each emotion in separate graphs
def plot_rms_values(rms_values, labels, test_rms=None):
    fig, axs = plt.subplots(len(emotion_classes), 1, figsize=(10, 15), sharex=True)

    for i, emotion in enumerate(emotion_classes):
        emotion_rms = [rms for rms, label in zip(rms_values, labels) if label == emotion]
        for idx, rms in enumerate(emotion_rms):
            axs[i].plot(rms, label=f'{emotion.capitalize()} RMS' if idx == 0 else "", color=f'C{i}')
        axs[i].set_title(f'RMS Values for {emotion.capitalize()} Emotions')
        axs[i].set_xlabel('Frame')
        axs[i].set_ylabel('RMS Value')
        axs[i].legend()

    if test_rms is not None:
        for ax in axs:
            ax.plot(test_rms, label='Test RMS', color='green')

    plt.tight_layout()
    plt.show()

plot_rms_values(rms_values, labels, test_rms=rms_value)

# Plotting the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=label_encoder.transform(emotion_classes))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()
