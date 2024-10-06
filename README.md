
# Amharic OCR - Neural Network from Scratch

This repository implements an Optical Character Recognition (OCR) system for **Amharic script**, trained on a neural network built entirely from scratch—without using any machine learning frameworks like TensorFlow or PyTorch. The system focuses on recognizing **36 Amharic characters**, converting them from hand-drawn images to corresponding textual outputs.

## Features

- **Neural Network from Scratch**: Built using only fundamental libraries such as NumPy, implementing layers, activation functions, forward and backward propagation, and training algorithms manually.
- **Amharic Character Classification**: The neural network is designed to classify and recognize 36 distinct characters from the Amharic script.
- **End-to-End OCR System**: From image preprocessing to neural network-based classification, this project offers a complete OCR pipeline.
- **Focus on Educational Value**: The project serves as a learning resource for those interested in understanding how neural networks work at a low level, without relying on pre-built frameworks.

## Project Structure

- **data/**: Contains training and test datasets for Amharic characters.
- **src/**: Implementation of the neural network model, activation functions, loss functions, and training scripts.
- **utils/**: Helper functions for data loading, preprocessing, and visualization.

## How it Works

1. **Data Preprocessing**: The input data consists of images of handwritten Amharic characters. These images are preprocessed and converted to a format suitable for input to the neural network.
2. **Neural Network Architecture**: The network uses a multi-layer perceptron (MLP) architecture, with:
   - Input layer corresponding to flattened image pixels.
   - Hidden layers with activation functions like ReLU or Sigmoid.
   - Output layer with 37 neurons (one for each character).
3. **Training Process**: The model is trained using backpropagation and gradient descent, optimizing a loss function for character classification accuracy.
4. **Evaluation**: The trained model is evaluated on a test dataset of Amharic characters, assessing its ability to accurately predict the correct letter from unseen inputs.

## Goals
- Build a fully functional OCR system for Amharic text.
- Gain a deep understanding of how neural networks are implemented from the ground up.
- Create a reusable and modifiable neural network implementation for other text recognition tasks.

## Future Work

- Expand the character set to include more Amharic letters and symbols.
- Improve the model by adding convolutional layers for better image feature extraction.
- Implement different training optimizers like Adam or RMSProp for faster convergence.

## Installation

To run this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/kablekid/amharic-ocr-from-scratch.git
cd amharic-ocr-from-scratch
pip install -r requirements.txt
```

Then, you can start training the neural network using:

```bash
python src/train.py
```
