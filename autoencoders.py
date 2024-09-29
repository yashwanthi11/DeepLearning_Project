import tensorflow as tf
from keras.src.layers import UpSampling2D
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Input, Conv2DTranspose, Reshape
from tensorflow.keras.optimizers import Adam

# Set up data generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Define image size (assuming your images are 224x224)
img_width, img_height = 224, 224

train_generator = train_datagen.flow_from_directory(
    directory='C:/Users/yaswa/PycharmProjects/DeepLearning/StaelliteImageTag/dataoftwo',
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='input'  # For autoencoders, set class_mode to 'input'
)

test_generator = test_datagen.flow_from_directory(
    directory='C:/Users/yaswa/PycharmProjects/DeepLearning/StaelliteImageTag/dataoftwo',
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='input'  # For autoencoders, set class_mode to 'input'
)

# Define the encoder
encoder_input = Input(shape=(img_width, img_height, 3))
encoder_conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
encoder_pool1 = MaxPooling2D((2, 2), padding='same')(encoder_conv1)
encoder_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(encoder_pool1)
encoder_pool2 = MaxPooling2D((2, 2), padding='same')(encoder_conv2)
encoder_conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(encoder_pool2)
encoder_output = MaxPooling2D((2, 2), padding='same')(encoder_conv3)

# Define the decoder
decoder_conv1 = Conv2D(128, (3, 3), activation='relu', padding='same')(encoder_output)
decoder_upsample1 = UpSampling2D((2, 2))(decoder_conv1)
decoder_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(decoder_upsample1)
decoder_upsample2 = UpSampling2D((2, 2))(decoder_conv2)
decoder_conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(decoder_upsample2)
decoder_upsample3 = UpSampling2D((2, 2))(decoder_conv3)
decoder_output = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(decoder_upsample3)  # 3 channels for RGB image

# Combine encoder and decoder into autoencoder
autoencoder = Model(encoder_input, decoder_output)

# Compile the model
autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

# Train the autoencoder
autoencoder.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=3,
    validation_data=test_generator,
    validation_steps=len(test_generator)
)

# Save the autoencoder model
autoencoder.save('autoencoder_model.h5')
