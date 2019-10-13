import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def baseline_cnn(image_height, image_width, channels, ):
    """baseline cnn for image classification
    from: https://www.tensorflow.org/tutorials/images/cnn
    """
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, channels)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))
    return model
