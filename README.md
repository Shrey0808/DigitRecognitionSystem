# Project Description: Handwritten Number Predictor using Convolutional Neural Network (CNN)

## Abstract:
The aim of this project is to predict handwritten numbers from image inputs using deep learning and computer vision techniques.
  
## Table of Contents:
1. Introduction
2. Dataset Understanding
3. Data Preprocessing
4. Model Architecture
5. Base model Training Phase
6. Base model Testing Phase
7. Final Number Prediction
8. Results
9. Conclusion

## 1. Introduction:
Handwritten digit recognition is a classic problem in computer vision and serves as a foundational task for understanding deep learning concepts. This project takes this problem to a whole new different level. My current model consists of :

- **Base Model: Digit Recognition System**
  - Trained using the MNIST dataset with 60,000 training images and 10,000 testing images.
  - Achieves an accuracy of 99.10% on the testing dataset.
  - Implemented with Convolutional Neural Networks (CNN) for effective feature extraction and classification.

- **Advanced Model: Computer Vision for Digit Segmentation**
  - Utilizes computer vision techniques to segment handwritten number images into individual digits.
  - Makes predictions for each segmented digit using the base CNN model.

## 2. Dataset Understanding :
The MNIST dataset contains 60,000 images to train the model and 10,000 images to test the accuracy of the model. Each image is of size (28x28) with each pixel value ranging from 0 to 255. 

## 3. Dataset Preprocessing:
Dataset preprocessing is essential to ensure that the data is in a suitable format and condition for training machine learning models. For this dataset, our preprocessing step will involve normalization and reshaping the data appropriately.

### Reshaping the Images:
The original MNIST dataset consists of grayscale images represented as 28x28 matrices. However, convolutional neural networks (CNNs) expect input images to have a depth dimension, even for grayscale images so we reshape our images from 28x28 to 28x28x1 where the `1` represents the channel for grayscale images.

### Normalizing Pixel Values
Normalizing pixel values is crucial to ensure consistent and stable model training. In the case of grayscale images in the MNIST dataset, pixel values range from 0 to 255. We normalize the pixel values in the range of 0 to 1 to make it easy for our model to understand and process them without changing the weightage of each pixel to our image


## 4. Model Architecture:
Certainly! Here's a brief explanation of the Convolutional Neural Network (CNN) architecture that you've defined for the MNIST handwritten digit classification task:

### CNN Architecture Overview

The CNN architecture of this model consists of several layers that are designed to extract and process features from the input images and make predictions.

1. **Input Layer**:
   - The first layer in the model is a convolutional layer (`Conv2D`) with 64 filters of size `(3, 3)`. This layer is responsible for learning features from the input images. The activation function used is ReLU (Rectified Linear Unit), which introduces non-linearity.
   - `input_shape` is set to `(28, 28, 1)` to match the dimensions of the input images after reshaping (`(28, 28, 1)` where `1` represents the grayscale channel).

2. **Pooling Layers**:
   - After each convolutional layer, a max pooling layer (`MaxPooling2D`) with a pool size of `(2, 2)` is applied. Max pooling reduces the spatial dimensions (width and height) of the input, thereby reducing the number of parameters and computations in the network while preserving important features.

3. **Additional Convolutional Layers**:
   - Two more convolutional layers follow, each with 32 filters of size `(3, 3)` and ReLU activation. These layers further extract higher-level features from the data.

4. **Flattening Layer**:
   - Before passing the extracted features to the fully connected layers, the output from the last convolutional layer is flattened (`Flatten`). This converts the 2D matrix representation of the features into a vector that can be fed into a dense neural network.

5. **Dense Layers (Fully Connected Layers)**:
   - The flattened output is then passed through two dense layers (`Dense`), each with 64 and 32 units respectively, both using ReLU activation functions. These layers perform classification based on the extracted features.
   
6. **Output Layer**:
   - The final layer (`Dense`) has 10 units corresponding to the 10 possible classes (digits 0-9 in MNIST). The activation function used is `softmax`, which outputs probabilities for each class, indicating the model's confidence in its predictions.

### Model Compilation

- **Loss Function**: Sparse categorical cross-entropy (`sparse_categorical_crossentropy`) is used as the loss function. It is suitable for integer-encoded target labels (like in MNIST).
  
- **Optimizer**: Adam optimizer (`'adam'`) is chosen for its adaptive learning rate and momentum properties, which typically lead to faster convergence during training.
  
- **Metrics**: The model is evaluated based on accuracy (`'accuracy'`), which measures the fraction of correctly classified images.

### Computer Vision Architecture

1. **Base layer Loading**
   - The trained Handwritten Digit Predictor Model is loaded which serves as the base layer for our project.
2. **Image input**
   - A User-Friendly website design is made to upload the images of handwritten numbers. The frontend of the website is designed using HTML/CSS and Javascript whereas the backend is built on Flask architecture.
3. **Image Preprocessing**
   - The input image is read using OpenCV and firstly converted to grayscale. Gaussian blur and adaptive thresholding is applied to enhance the digits extraction. Finally the contours of the digits are found.


## 5. Base model Training Phase:
The model is trained using the Adam optimizer and sparse_categorical_crossentropy loss. The training loop iterates through the MNIST training dataset, updating the model's weights to minimize the classification loss. Training progress is monitored through the display of loss values per epoch.

## 6. Base model Testing Phase:
The trained model is evaluated on a separate test set to assess its generalization performance. The accuracy on the test set is calculated, demonstrating the model's ability to accurately classify unseen handwritten digits.

## 7. Final Number Prediction
The model iterates through the identified contours, extracting each digit and preprocessing it for model prediction. The extracted digits are resized to 28x28x1 pixels and then normalized for model input. Each extracted digit is then passed to the loaded model to predict the class label.

## 8. Results:
The trained CNN achieves a high accuracy of 99.10% on the MNIST test set, showcasing its effectiveness in recognizing handwritten digits.

## 9. Conclusion
This project serves as a valuable resource for understanding the implementation of a CNN for handwritten number recognition and CV for digit extraction, making it suitable for both educational purposes and practical applications.
