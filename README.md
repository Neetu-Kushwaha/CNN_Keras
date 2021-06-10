# CNN_Keras
Prerequisites
•	Python 3.6+
•	PyTorch 1.0+
•	Keras
•	Sklearn

Getting started
CIFAR-10 Dataset:-
The CIFAR-10 dataset consists of 60000 32×3232×32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
Data Storage Format
Pickle module is used to store the data. Python pickle module is used for serializing and de-serializing a Python object structure. Any object in Python can be pickled so that it can be saved on disk. What pickle does is that it “serializes” the object first before writing it to file. Pickling is a way to convert a python object (list, dict, etc.) into a character stream. The idea is that this character stream contains all the information necessary to reconstruct the object in another python script.

KerasCNN.ipynb Notebook:-
In this notebook, we will learn to define a CNN for classification of the CIFAR-10 dataset using Keras and store activation values of intermediate layers of testing data.
Training: Training an image classifier
We will do the following steps in order:
1.	Load and normalizing the CIFAR10 training and test datasets using Keras.
2.	Define a Convolutional Neural Network
3.	Define a loss function
4.	Train the network on the training data
5.	Test the network on the test data
1.	Store Activation values of the intermediate hidden layer for each test image. The activation dataset consists of two binary classes-Positive Class (CNN makes a wrong prediction (Mistake)) and Negative class (CNN makes a correct prediction).
Test Accuracy:
The test accuracy reached 64%.
Output:
- CNN Keras model for CIFAR-10 dataset.
- Activation dataset and prediction Class (Python Pickle Module is used to save the data/file)

