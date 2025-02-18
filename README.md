# NeuroDebug: Neural Network Debugging and Activation Analysis

## Overview
NeuroDebug is a deep learning framework designed for **debugging convolutional neural networks (CNNs)** by analyzing activations and diagnosing misclassifications. This project is built using **TensorFlow/Keras** and focuses on the **CIFAR-10 dataset**.

## Features
- **Train & Evaluate CNN** on the CIFAR-10 dataset.
- **Predict** test images and analyze model decisions.
- **Extract Activation Maps** of intermediate layers.
- **Debug Misclassifications** by tracking activation patterns.
- **Store and Visualize Activation Data** to improve interpretability.

## Prerequisites
Ensure you have the following installed:
- Python **3.6+**
- TensorFlow/Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Pickle

You can install dependencies using:
```sh
pip install tensorflow numpy pandas scikit-learn matplotlib
```

## Getting Started
### CIFAR-10 Dataset
The CIFAR-10 dataset consists of **60,000 32×32 color images** in 10 classes, with 6,000 images per class. The dataset is divided into **50,000 training images** and **10,000 test images**.

### Data Storage Format
The **Pickle module** is used for storing the dataset. Python's Pickle module serializes and deserializes Python objects, allowing data to be saved and retrieved efficiently.

### KerasCNN.ipynb Notebook
This Jupyter notebook provides an in-depth **step-by-step implementation** of a CNN for CIFAR-10 classification using Keras. It includes:
1) Loading and normalizing the CIFAR-10 dataset.
2) Defining a **Convolutional Neural Network**.
3) Specifying a **loss function**.
4) Training the model on CIFAR-10 images.
5) Evaluating model performance on test data.
6) **Storing activation values** of intermediate layers for debugging misclassifications.

#### Test Accuracy
The trained model achieved a **test accuracy of 64%**.

### Output
- **CNN Keras model** for CIFAR-10 classification.
- **Activation dataset and predictions**, stored using Python Pickle.

### 1. Train the Model
Run the following command to train the CNN:
```sh
python main.py --train
```

### 2. Evaluate the Model
To evaluate test accuracy, use:
```sh
python main.py --evaluate
```

### 3. Predict an Image
For making predictions, provide an image path:
```sh
python main.py --predict path/to/image.jpg
```

### 4. Extract and Debug Activations
Activation values for test images are stored and analyzed. The activation dataset is saved in `Activation_dataset.p`, and misclassified images are tracked in `classes_activation.p`.

## Model Architecture
The CNN consists of:
- **Convolutional Layers** for feature extraction
- **Pooling Layers** to reduce dimensionality
- **Fully Connected Layers** for classification

## Debugging with Activation Maps
The model extracts activation values from intermediate **Pooling** and **Dense** layers to analyze how CNNs process images. Misclassified images are identified and stored for further analysis.

## Output Files
- **`Activation_dataset.p`** → Stores activation values of intermediate layers.
- **`classes_activation.p`** → Contains class predictions and misclassified indices.

## Contribution
Feel free to fork this repository and contribute to improving neural network debugging!

## License
MIT License
