# Speech Emotion Recognition
Speech Emotion Recognition using MFCC, LPC, and STFT features with SVM and DCNN classifiers. Includes data loading, feature extraction, model training, evaluation, and visualization. Easily extensible for new datasets and experiments.

## Overview
This project implements various pipelines for speech emotion recognition using different feature extraction methods (MFCC, LPC, STFT) and classifiers (SVM, DCNN). The goal is to accurately classify emotions from speech audio files.

## Methodology
This project explores multiple combinations of feature extraction and classification techniques for speech emotion recognition:

### 1. MFCC + SVM (`mfcc_svm.py`)
- **Feature Extraction:** Mel-Frequency Cepstral Coefficients (MFCCs) are extracted from each audio file, along with RMS energy and pitch features. These features are combined into a single feature vector.
- **Classification:** A Support Vector Machine (SVM) classifier is trained on the extracted features to predict the emotion label.

### 2. MFCC + DCNN (`mfcc_dcnn.py`)
- **Feature Extraction:** MFCCs, RMS, and pitch features are extracted as above.
- **Classification:** A Deep Convolutional Neural Network (DCNN) is used to learn from the feature vectors and classify emotions.

### 3. LPC + SVM (`lpc_svm.py`)
- **Feature Extraction:** Linear Predictive Coding (LPC) coefficients are extracted from each audio file, along with RMS and pitch features. Features are padded or truncated to a fixed length.
- **Classification:** An SVM classifier is trained on the LPC-based feature vectors for emotion recognition.

### 4. LPC + DCNN (`dcnn_lpc.py`)
- **Feature Extraction:** LPC, RMS, and pitch features are extracted and padded to a consistent length.
- **Classification:** A DCNN is trained on the padded LPC feature matrices to classify emotions.

### 5. STFT + SVM (`stft_svm.py`)
- **Feature Extraction:** Short-Time Fourier Transform (STFT) magnitude spectra are computed for each audio file, along with RMS and pitch features. Features are flattened and padded as needed.
- **Classification:** An SVM classifier is trained on the STFT-based feature vectors.

### 6. STFT + DCNN (`dcnn_stft.py`)
- **Feature Extraction:** STFT, RMS, and pitch features are extracted and padded to a fixed length.
- **Classification:** A DCNN is trained on the 2D STFT feature matrices for emotion classification.

All approaches use label encoding for emotion categories and split the dataset into training and testing sets for evaluation. Model performance is measured using accuracy and precision metrics, and results are visualized with plots of training history, RMS, and pitch statistics.

## Project Structure
- `src/` - Source code for feature extraction and model training/evaluation
- `tests/` - Unit tests
- `sad_fear_test_file.wav` - Example audio file for testing
- `project_report.docx` - Project documentation
