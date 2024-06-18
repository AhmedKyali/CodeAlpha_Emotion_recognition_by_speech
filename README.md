# Emotion Recognition from Speech Audio

This project implements a deep learning model to recognize emotions in speech audio. The model classifies spoken sentences into various emotions such as happiness, anger, sadness, and more. This document provides an overview of the project, the steps followed, and instructions for running the code.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Requirements](#requirements)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Overview
The goal of this project is to develop a deep learning model that can accurately classify emotions from speech audio clips. We use mel-spectrograms as features for the model and a Convolutional Neural Network (CNN) for classification.

## Dataset
The dataset used in this project is the Toronto Emotional Speech Set (TESS). Each audio file in the dataset is labeled with one of the following emotions:
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Pleasant Surprise (ps)
- Sad

## Preprocessing
1. **Load Data:** The audio files are loaded, and the corresponding labels are extracted.
2. **Mel-Spectrogram Generation:** Each audio file is converted into a mel-spectrogram.
3. **Normalization:** The mel-spectrograms are normalized using z-score normalization.
4. **Padding:** The spectrograms are padded to ensure uniform shape for input into the model.

## Model Architecture
The model is a Convolutional Neural Network (CNN) with the following layers:
1. **Input Layer:** Shape of (128, 87), corresponding to the dimensions of the mel-spectrograms.
2. **Convolutional Layers:** Four Conv1D layers with varying filter sizes and ReLU activation, followed by MaxPooling and Dropout layers.
3. **Flatten Layer:** Converts the 2D feature maps into a 1D feature vector.
4. **Dense Layers:** Two fully connected layers with ReLU activation and dropout for regularization.
5. **Output Layer:** A softmax layer with the number of neurons equal to the number of emotion classes.

## Training
The model is compiled using categorical crossentropy loss and the Adam optimizer. Early stopping and model checkpointing are used to prevent overfitting and save the best model.

```python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)
mc = callbacks.ModelCheckpoint('best_model.keras', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

history = model.fit(x_train, y_train, epochs=100, callbacks=[es, mc], validation_data=(x_test, y_test), batch_size=32)
```

## Evaluation
The model's performance is evaluated using accuracy and loss metrics on the training and validation sets. The training history is visualized using plots.

## Requirements
- Python 3.x
- NumPy
- Pandas
- TensorFlow / Keras
- Matplotlib
- Seaborn
- Librosa
- Scikit-learn

Install the required packages using:

```bash
pip install numpy pandas tensorflow matplotlib seaborn librosa scikit-learn
```

## Usage
1. **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2. **Prepare the dataset:**
    - Ensure the dataset is available in the correct directory.
    - Update the dataset path in the code if necessary.
3. **Run the script:**
    ```bash
    python emotion_recognition.py
    ```
4. **Evaluate the model:**
    - The trained model is saved as `best_model.keras`.
    - Use the saved model to make predictions on new audio files.

## Acknowledgements
This project uses the TESS dataset for emotion recognition in speech. We acknowledge the creators of the dataset and the libraries used in this project.

Feel free to reach out if you have any questions or suggestions.
