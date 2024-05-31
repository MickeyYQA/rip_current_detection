# rip_current_detection
A rip current detection system using the Tello drone and an Artificial Neural Network (ANN) model.

## Overview
This project implements a rip current detection system using the Tello drone and an Artificial Neural Network (ANN) model. The system captures real-time video feeds from the drone, preprocesses the images, and predicts the presence of rip currents with enhanced accuracy. The results are displayed in a user-friendly interface.

## Features
- **Real-time Image Transmission**: Establishes UDP communication to stream live video feed from the Tello drone's camera to the computer.
- **Image Preprocessing**: Uses OpenCV to resize the received images to 10x10 pixels and convert them to grayscale, ensuring consistency with the training data.
- **ANN Model for Rip Current Detection**: Utilizes a Multi-Layer Perceptron (MLP) Classifier from the Scikit-Learn library, configured with one hidden layer of 60 neurons, achieving an overall accuracy of 90.4977%.
- **Enhanced Accuracy Mechanism**: Predicts using the model 100 times per image and confirms the presence of a rip current if over 82% of the predictions indicate a rip current, improving the reliability of the detection.
- **Visual Result Display**: Displays detection results in a window, showing "RIP CURRENT" with a red background if a rip current is detected and "SAFE" with a green background if no rip current is detected.


## Prerequisites
- Python 3.x
- OpenCV
- Scikit-Learn
- Pandas
- Numpy
- Djitellopy
- Tello Talent Robomaster TT drone

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/MickeyYQA/rip-current-detection.git
    cd rip-current-detection
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Dataset Preparation
1. Collect and save images of near-shore conditions, including both rip currents and non-rip currents.
2. Use the provided script to preprocess and convert images to a CSV file. See: grayingAndSizing.py and createCSV.py

## Model Training
Load the dataset from the CSV file and train the model. See: rip-model-train.py

## Real-time Detection
Implement the real-time detection system with the Tello drone. See: rip-detect.py
