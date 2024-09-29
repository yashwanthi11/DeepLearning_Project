import os
import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import preprocess as dp

if __name__ == "__main__":
    images_folder_path = 'data'
    imdata = dp.PreProcess_Data()
    imdata.visualization_images(images_folder_path, 2)
    project_df, train, label = imdata.preprocess(images_folder_path)

    # Map labels to 0 and 1 for binary classification
    label_mapping = {'class_1': 0, 'class_2': 1}  # Update with your actual class names
    project_df['Encoded_Labels'] = project_df['Labels'].map(label_mapping).astype(str)

    # Split the dataset into training and testing sets
    train_df, test_df = train_test_split(project_df, test_size=0.2, random_state=42)

    # Choose a preferred optimizer (e.g., 'adam')
    chosen_optimizer = 'adam'

    # ImageDataGenerator for data augmentation and normalization
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
                                                        x_col="Image",
                                                        y_col="Encoded_Labels",
                                                        target_size=(224, 224),
                                                        batch_size=32,
                                                        class_mode='categorical')

    test_generator = test_datagen.flow_from_dataframe(dataframe=test_df,
                                                      x_col="Image",
                                                      y_col="Encoded_Labels",
                                                      target_size=(224, 224),
                                                      batch_size=32,
                                                      class_mode='categorical')

    # Build the sequential model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(224, activation='relu'))

    # Output layer with 2 neurons for binary classification
    model.add(Dense(1, activation='softmax'))

    # Compile the model with the chosen optimizer and appropriate loss and metrics
    model.compile(optimizer=chosen_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(train_generator, epochs=3, validation_data=test_generator)

    # Save the model with a clear and informative filename
    model_filename = 'best_model.h5'  # Consider saving based on validation performance
    model.save(model_filename)

    # Plot training history
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.show()
