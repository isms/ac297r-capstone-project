import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def baseline_cnn(image_dim = (100, 100, 5)):
    """baseline cnn for image classification
    from: https://www.tensorflow.org/tutorials/images/cnn
    """
    image_height, image_width, channels = image_dim
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=image_dim))
    model.add(layers.Dropout(0.2))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def combined_cnn(gsv_image_dim = (300, 300, 3), sat_image_dim = (100, 100, 5), n_classes = 3):
    """Combined CNN using satellite and Street View imagery.
    """

    # 1. Google Street View (GSV) Image Input
    gsv_input_img = keras.Input(shape=gsv_image_dim, name='image_input')

    gsv_cnn = layers.Conv2D(64, (3,3), activation = 'relu')(gsv_input_img)
    gsv_cnn = layers.MaxPooling2D((2,2))(gsv_cnn)
    gsv_cnn = layers.Conv2D(32, (3,3), activation = 'relu')(gsv_cnn)
    gsv_cnn = layers.MaxPooling2D((2,2))(gsv_cnn)
    gsv_cnn = layers.Conv2D(32, (3,3), activation = 'relu')(gsv_cnn)
    gsv_flat = layers.Flatten()(gsv_cnn)
    gsv_image_embedding = Dense(100, activation='relu')(gsv_flat)

    # 2. Satellite Image input
    sat_input_img = keras.Input(shape=sat_image_dim, name='image_input')

    sat_cnn = layers.Conv2D(64, (3,3), activation = 'relu')(sat_input_img)
    sat_cnn = layers.MaxPooling2D((2,2))(gsv_cnn)
    sat_cnn = layers.Conv2D(32, (3,3), activation = 'relu')(sat_cnn)
    sat_cnn = layers.MaxPooling2D((2,2))(sat_cnn)
    sat_cnn = layers.Conv2D(32, (3,3), activation = 'relu')(sat_cnn)
    sat_flat = layers.Flatten()(sat_cnn)
    sat_image_embedding = Dense(100, activation='relu')(sat_flat)

    # 3. Concatenate embeddings
    concat = layers.Concatenate(axis=1)([gsv_image_embedding, sat_image_embedding])

    # 4. Dense layers + classification
    full_embedding = Dense(300, activation='relu')(concat)
    full_embedding = Dense(100, activation='relu')(full_embedding)
    output = Dense(n_classes, activation='softmax')(full_embedding)

    # 5. define full model and compile
    model = models.model(inputs=[gsv_input_img, sat_image_dim], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
