import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from keras.src.legacy.preprocessing.image import ImageDataGenerator


class CNN_Model:
    def create_model(self, input_shape, num_classes):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def plot_history(self, history):
        # Plot training & validation accuracy values
        plt.figure(figsize=(12, 6))

        # Plot accuracy subplot
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, index 1
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'], loc='upper left')

        # Plot loss subplot
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, index 2
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.tight_layout()
        plt.show()

    def save_model(self, model, filepath):
        model.save(filepath)
        print("Model saved successfully.")


# Create an instance of CNN_Model
cnn_model = CNN_Model()

# Assuming you have defined input_shape and num_classes previously
input_shape = (224, 224, 3)  # Example input shape
num_classes = 2  # Example number of classes

# Create the model
model = cnn_model.create_model(input_shape, num_classes)

# Save the model
cnn_model.save_model(model, 'cnn_model.h5')
