from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.regularizers import l2

class CNN_Model:
    @staticmethod
    def create_regularized_model(input_shape, num_classes):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01), input_shape=input_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))  # Dropout layer to reduce overfitting
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))  # Dropout layer to reduce overfitting
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))  # Dropout layer to reduce overfitting
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    @staticmethod
    def save_model(model, filepath):
        model.save(filepath)
        print("Model saved successfully.")

# Create an instance of CNN_Model
cnn_model = CNN_Model()

# Assuming you have defined input_shape and num_classes previously
input_shape = (224, 224, 3)  # Example input shape
num_classes = 2  # Example number of classes

# Create the regularized model
regularized_model = cnn_model.create_regularized_model(input_shape, num_classes)

# Save the regularized model
CNN_Model.save_model(regularized_model, 'regularized_cnn_model.h5')
