# Micro-Facial-Expression-Detection (Lie,Truth) Classification

![dataset-cover](https://github.com/user-attachments/assets/739ef6a4-59ef-44f9-9576-1931d8fe7a62)

**Micro-Facial Expression Detection using CNN cassification for subtle emotion recognition from facial movements.**

A deep learning pipeline for detecting micro-facial expressions using Convolutional Neural Networks (CNN) techniques. This project aims to recognize subtle, involuntary facial expressions that are difficult to detect with the naked eye, enabling applications in emotion analysis, security, and human-computer interaction.

Orignal Dataset --> Micro Expression Dataset for Lie Detection https://www.kaggle.com/datasets/devvratmathur/micro-expression-dataset-for-lie-detection 

By: Devvrat Mathur

Edited Dataset --> https://www.kaggle.com/datasets/mohammadabuayyash1/micro-data By: Mohammad Abu Ayyash

Micro expressions can occur either as the result of conscious suppression or unconscious repression of emotions. As such, spotting micro expressions (or analyzing facial expressions) is key to learning how to detect lies, unveiling concealed emotions, and deception.

## What This Notebook Does

- Loads preprocessed NumPy arrays:

- Normalizes the data to the [0, 1] range
- Converts labels to categorical format using `LabelEncoder` and `to_categorical`
- Splits data into training and testing sets
- Defines a CNN model using Keras:
  - 3 convolutional blocks (Conv2D → MaxPooling2D → Dropout)
  - Flatten → Dense → Dropout → Dense (softmax output)
- Compiles the model using categorical crossentropy and Adam optimizer
- Trains the model for 10 epochs with batch size 32
- Evaluates the model on test data and prints accuracy and classification report

## Requirements

- Python
- TensorFlow / Keras
- NumPy
- scikit-learn
- Matplotlib

## Author

Notebook by **Mohammad Abu Ayyash** 

Also on Kaggle:
[[Kaggle notebook](ht](https://www.kaggle.com/code/mohammadabuayyash1/micro-facial-expression-detection)
