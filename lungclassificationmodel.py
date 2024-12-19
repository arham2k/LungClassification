

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import seaborn as sns
from sklearn.metrics import confusion_matrix

dataset_path = "./COVID-19_Radiography_Dataset"
categories = ["COVID", "NORMAL", "PNEUMONIA"]

print(dataset_path)
print(categories)

# Function to load sample images
def load_sample_images(category):
    category_path = os.path.join(dataset_path, category)
    if not os.path.exists(category_path):
        print(f"Category path not found: {category_path}")
        return []
    filenames = os.listdir(category_path)[:5]  # Load the first 5 images, os.listdir: Lists files in a directory.
    print(f"Filenames in {category}: {filenames}")  # Debugging line
    images = [cv2.imread(os.path.join(category_path, file)) for file in filenames if file.endswith(('.png', '.jpg', '.jpeg'))] # cv2.imread: Reads an image into a NumPy array.
    return images

# Visualize images
def visualize_images(images, category):
    plt.figure(figsize=(12, 6))
    for i, img in enumerate(images):
        plt.subplot(1, 5, i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # cv2.cvtColor: Converts images from BGR (default for OpenCV) to RGB.
        plt.title(category)
        plt.axis("off")
    plt.show()

# Categories to load
categories = ["COVID", "NORMAL", "PNEUMONIA"]
for category in categories:
    images = load_sample_images(category)
    if not images:
        print(f"No images found in category: {category}")
        continue
    visualize_images(images, category)

#Pre Processing Data
def preprocess_image(image, target_size=(128, 128)):
    # Convert to grayscale (optional)
    # This converts the color image to grayscale. Grayscale images reduce the number of input channels from 3 (RGB) to 1.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resizes the image to a fixed size (128x128 in this case). This ensures that every image input has the same dimensions, which is essential for feeding them into the neural network.
    # Resize image to the target size (e.g., 128x128)
    image = cv2.resize(image, target_size)

    #Pixel values in an image typically range from 0 to 255. Normalizing them by dividing by 255 scales the values to a range between 0 and 1. This helps the model converge faster during training.
    # Normalize pixel values to be between 0 and 1
    image = image / 255.0

    #  Adds an extra dimension to the image, converting it into a 4D tensor that represents the batch size, height, width, and number of channels.
    # This is required for deep learning frameworks like TensorFlow/Keras
    # Expand dimensions to match model input (1, 128, 128, 1)
    image = np.expand_dims(image, axis=-1)

    return image

# Function to load and preprocess all images in the dataset
def load_data(dataset_path, categories, target_size=(128, 128)):
    images = []
    labels = []

    # Loop through each category (COVID, NORMAL, PNEUMONIA)
    for label, category in enumerate(categories):
        category_path = os.path.join(dataset_path, category)
        filenames = os.listdir(category_path)

        # Loop through all the images in the category
        for file in filenames:
            # Load the image
            image_path = os.path.join(category_path, file)
            image = cv2.imread(image_path)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue  # Skip this image and move to the next one

            # Preprocess the image
            image = preprocess_image(image, target_size)

            # Append to our images and labels lists
            images.append(image)
            labels.append(label)  # 0 for COVID, 1 for NORMAL, 2 for PNEUMONIA

    # Convert lists to numpy arrays for model training
    images = np.array(images)
    labels = np.array(labels)

    # Split the dataset into training, validation, and test sets (80%, 10%, 10%)
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# Load and preprocess the data
(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(dataset_path, categories)

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Test data shape: {X_test.shape}")

# Model Construction
import tensorflow as tf
from tensorflow.keras import layers, models

# Build the CNN model
def build_model(input_shape=(128, 128, 1)):
    model = models.Sequential()

    # Convolutional Layer 1
    # Adds a convolutional layer with 32, 64, or 128 filters, each of size (3x3). The relu activation is applied after the convolution to introduce non-linearity.
    # MaxPooling2D: Performs max pooling with a pool size of (2x2), which reduces the spatial dimensions by half
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional Layer 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional Layer 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten the 3D feature maps to 1D
    # Flatten: Converts the 3D tensor (after convolution and pooling) into a 1D vector to feed into the fully connected layer.
    model.add(layers.Flatten())

    # Fully Connected Layer (Dense)
    # Dense: A fully connected layer that performs the final classification.
    model.add(layers.Dense(128, activation='relu'))

    # softmax: Ensures that the output of the model represents class probabilities, useful for multi-class classification.
    # Output Layer (Softmax for multi-class classification)
    model.add(layers.Dense(3, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Train Model

# Build the CNN model
model = build_model()

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Define a function to predict the label of a single image
def predict_image(image_path, model, categories):
    # Load the image
    image = cv2.imread(image_path)

    # Preprocess the image (same preprocessing as training data)
    preprocessed_image = preprocess_image(image, target_size=(128, 128))
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Add batch dimension

    # Predict the label using the trained model
    predictions = model.predict(preprocessed_image)

    # Get the class with the highest probability
    predicted_label = np.argmax(predictions)
    predicted_class = categories[predicted_label]

    # Return the result
    return predicted_class, predictions


# Path to the test image
test_image_path = "COVID-19_Radiography_Dataset/NORMAL/Normal-1.png"

# Predict the label of the test image
predicted_class, probabilities = predict_image(test_image_path, model, categories)



# Display the result
print(f"Predicted Class: {predicted_class}")
print(f"Class Probabilities: {probabilities}")

# Function to display image with prediction
def display_prediction(image_path, predicted_class):
    # Load and display the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib

    plt.imshow(image_rgb)
    plt.title(f"Predicted Class: {predicted_class}")
    plt.axis("off")
    plt.show()


def plot_probabilities(probabilities, categories):
   # Average the probabilities across all images
    avg_probabilities = np.mean(all_probabilities, axis=0)

    # Plot the averaged probabilities for each class
    plt.bar(categories, avg_probabilities, color="skyblue")
    plt.ylabel("Average Probability")
    plt.title("Average Class Probabilities for All Images")
    plt.ylim(0, 1)
    plt.show()

# Directory containing the images (assumes subfolders for each class)
image_dir = "COVID-19_Radiography_Dataset\Test"
true_labels = []
predicted_labels = []
all_probabilities = []


# Loop through each class in the image directory
for class_name in os.listdir(image_dir):
    class_path = os.path.join(image_dir, class_name)

    # Check if it's a directory (to avoid files in the main directory)
    if os.path.isdir(class_path):
        # Loop through all images in the subdirectory
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)

            # Make sure the file is an image (you can check for extensions like .png, .jpg, etc.)
            if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Predict the label of the image
                predicted_class, probabilities = predict_image(image_path, model, categories)

                # Display the result
                print(f"Image: {image_name}")
                print(f"Predicted Class: {predicted_class}")
                print(f"Class Probabilities: {probabilities}")
                print("-" * 50)

                # Display the image with the prediction
                #  display_prediction(image_path, predicted_class)

                # Store the true label (we assume the class_name is the true label)
                true_labels.append(class_name)
                predicted_labels.append(predicted_class)

                # Plot the probabilities
                all_probabilities.append(probabilities[0])

# After all predictions, plot the average probabilities for each class
plot_probabilities(all_probabilities, categories)

predicted_labels_int = [0 if label.lower() == 'covid' else 1 if label.lower() == 'normal' else 2 for label in predicted_labels]
true_labels_int = [0 if label.lower() == 'covid' else 1 if label.lower() == 'normal' else 2 for label in true_labels]

from sklearn.metrics import classification_report
print(classification_report(true_labels_int, predicted_labels_int))

# Create confusion matrix
cm = confusion_matrix(true_labels_int, predicted_labels_int, labels=[0, 1, 2])

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

"""## Image with Predicted Label and Confidence Score"""

def visualize_prediction_with_score(image_path, predicted_class, probabilities):
    # Load and convert the image to RGB
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Add prediction text with OpenCV
    confidence = np.max(probabilities)  # Highest probability
    text = f"{predicted_class}: {confidence:.2f}"
    image_with_text = cv2.putText(
        image_rgb.copy(), text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA
    )

    # Display the image
    plt.imshow(image_with_text)
    plt.title("Prediction with Confidence Score")
    plt.axis("off")
    plt.show()

# Call the function
visualize_prediction_with_score(test_image_path, predicted_class, probabilities[0])

"""## Interactive Visualization with Jupyter Widgets"""

import ipywidgets as widgets
from IPython.display import display

def interactive_prediction(model, categories):
    uploader = widgets.FileUpload(accept="image/*", multiple=False)
    display(uploader)

    def on_upload(change):
        if uploader.value:
            # Read and predict uploaded image
            file_info = next(iter(uploader.value.values()))
            image_path = "/tmp/temp_image.jpg"
            with open(image_path, "wb") as f:
                f.write(file_info["content"])

            predicted_class, probabilities = predict_image(image_path, model, categories)
            display_prediction(image_path, predicted_class)
            print(f"Predicted Class: {predicted_class}")
            print(f"Class Probabilities: {probabilities}")

    uploader.observe(on_upload, names="value")

# Call the interactive function
interactive_prediction(model, categories)