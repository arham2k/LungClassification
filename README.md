# Lung Classification Model

This project aims to classify chest X-ray images into three categories: COVID-19, Normal, and Pneumonia. The classification model is built using TensorFlow and Keras, with Convolutional Neural Networks (CNNs) for image processing. The dataset used in this project is the COVID-19 Radiography Dataset, which contains labeled chest X-ray images for each class.

## Dataset

The dataset used in this project is the [COVID-19 Radiography Dataset](https://www.kaggle.com/datasets/andrewmvd/covid19-radiography-database), which contains images of chest X-rays from three classes:
- **COVID**: Images of patients infected with COVID-19.
- **NORMAL**: Images of healthy individuals.
- **PNEUMONIA**: Images of individuals diagnosed with pneumonia.

The dataset is divided into subfolders for each class, containing images of different formats (.jpg, .png, .jpeg).

## Project Overview

This project involves building a Convolutional Neural Network (CNN) for the classification of chest X-ray images. The following steps outline the workflow:

1. **Data Loading and Preprocessing**: 
    - Load images from each class.
    - Preprocess images by resizing them to a fixed size (128x128), converting them to grayscale, and normalizing pixel values.
    
2. **Model Building**: 
    - A CNN model is used with convolutional layers followed by pooling layers to extract features from the images.
    - A fully connected dense layer is used for final classification, outputting probabilities for each of the three classes using a softmax activation function.
    
3. **Model Training**: 
    - The model is trained using the dataset with a validation set for monitoring performance.
    
4. **Prediction and Evaluation**:
    - The trained model is used to make predictions on new chest X-ray images, and the results are displayed with the predicted class and confidence score.
    - The model's performance is evaluated using a confusion matrix and a classification report.
    
5. **Interactive Prediction**: 
    - A feature is included for interactive predictions, where users can upload chest X-ray images and get predictions from the trained model.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- Matplotlib
- Seaborn
- Scikit-learn
- ipywidgets (for interactive prediction in Jupyter)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/lung-classification-model.git
    cd lung-classification-model
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset:
    - Download the [COVID-19 Radiography Dataset](https://www.kaggle.com/datasets/andrewmvd/covid19-radiography-database) and place it in the `COVID-19_Radiography_Dataset` folder in the project directory.


