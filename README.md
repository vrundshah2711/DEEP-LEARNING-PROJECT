# DEEP-LEARNING-PROJECT

COMPANY: CODTECH IT SOLUTIONS

NAME: SHAH VRUND SNEHALKUMAR

INTERN ID: CT04DN1015

DOMAIN: DATA SCIENCE

DURATION: 4 WEEKS

MENTOR: NEELA SANTOSH

Objective:

The goal of Task 2 was to design, implement, and evaluate a deep learning model capable of classifying images into multiple categories using a convolutional neural network (CNN). For this task, the publicly available CIFAR-10 dataset was used, which is a benchmark dataset widely used in computer vision. The task required us to demonstrate a complete deep learning workflow including dataset loading, preprocessing, model construction, training, evaluation, and visualization of predictions. TensorFlow and Keras were used as the core frameworks due to their accessibility and industry relevance.

Dataset Overview:

CIFAR-10 is a collection of 60,000 color images divided into 10 different classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Each image is 32×32 pixels in RGB format. The dataset is split into 50,000 training images and 10,000 test images. For this project, to reduce training time and resource consumption, we used a smaller subset of 10,000 training samples and 2,000 test samples.

The dataset is built into TensorFlow, so no manual downloading was required. After loading, the images were normalized by dividing pixel values by 255.0 to bring them into the range [0, 1], which helps speed up convergence during model training.

Data Preprocessing:

Preprocessing steps included:

Normalization: All pixel values were scaled from 0–255 to a normalized range of 0–1.

Label Handling: CIFAR-10 labels are provided as arrays with shape (1,), which were reshaped using .item() for compatibility with plotting and display functions.

Visualization: A set of 10 sample images from the training data was displayed to help visualize the class distribution and data quality.

Model Architecture:

We built a sequential CNN model using TensorFlow's Keras API. The architecture included:

An input layer using tf.keras.Input

Three convolutional layers with ReLU activation

Two max-pooling layers to reduce dimensionality

A flatten layer to convert the 2D output into a 1D vector

A dense hidden layer with 64 neurons and ReLU activation

A final dense output layer with 10 neurons and softmax activation for multi-class classification

This architecture is well-suited for small image classification tasks and balances accuracy with computational efficiency.

Training and Evaluation:

The model was compiled using the Adam optimizer and sparse categorical crossentropy as the loss function. It was trained for 10 epochs on the reduced dataset. Training history, including accuracy and validation accuracy over epochs, was visualized using matplotlib.

Evaluation was done on the test set, and the final test accuracy was printed. On the subset, the model typically achieved over 65–70% accuracy, depending on runtime hardware and training variation.

Model Predictions:

We predicted the classes of a few sample test images and visualized them with matplotlib. Each image was displayed alongside its predicted and actual label. This helped us qualitatively assess the model’s performance and understand any common misclassifications.

Model Saving:

The final trained model was saved using the new Keras format (.keras), as recommended by the latest TensorFlow version. This allows the model to be reloaded in future notebooks or production environments without retraining.

Conclusion:

This task successfully demonstrated a complete deep learning pipeline for image classification using TensorFlow and the CIFAR-10 dataset. Through this task, we learned how to:

Load and preprocess image data

Build and compile a CNN using Keras

Train and evaluate the model on real-world image data

Visualize training results and predictions

Save the trained model for future use

This project is foundational for more advanced tasks such as transfer learning, real-time image detection, and deployment to web or mobile apps. The techniques applied here form a core skill set for any data science or machine learning role involving computer vision.
