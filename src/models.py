import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers

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

def combined_cnn(gsv_image_dim = (640, 640, 3), sat_image_dim = ((2100, 2100, 4)), n_classes = 2, loss_fn = 'binary_crossentropy'):

    """
    Combined CNN using satellite and Street View imagery.
    """
    # functions from http://www.deepideas.net/unbalanced-classes-machine-learning/
    def sensitivity(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + K.epsilon())

    def specificity(y_true, y_pred):
        true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
        possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
        return true_negatives / (possible_negatives + K.epsilon())

    # 1. Google Street View (GSV) Image Input
    gsv_input_img = layers.Input(shape=gsv_image_dim, name='gsv_image_input')

    gsv_cnn = layers.Conv2D(128, (3,3), activation = 'relu')(gsv_input_img)
    gsv_cnn = layers.MaxPooling2D((2,2))(gsv_cnn)
    gsv_cnn = layers.Conv2D(64, (3,3), activation = 'relu')(gsv_cnn)
    gsv_cnn = layers.MaxPooling2D((2,2))(gsv_cnn)
    gsv_cnn = layers.Conv2D(32, (3,3), activation = 'relu')(gsv_cnn)
    gsv_flat = layers.Flatten()(gsv_cnn)
    gsv_image_embedding = layers.Dense(200, activation='relu')(gsv_flat)

    # 2. Satellite Image input
    sat_input_img = layers.Input(shape=sat_image_dim, name='aerial_image_input')

    sat_cnn = layers.Conv2D(128, (3,3), activation = 'relu')(sat_input_img)
    sat_cnn = layers.MaxPooling2D((2,2))(sat_cnn)
    sat_cnn = layers.Conv2D(64, (3,3), activation = 'relu')(sat_cnn)
    sat_cnn = layers.MaxPooling2D((2,2))(sat_cnn)
    sat_cnn = layers.Conv2D(32, (3,3), activation = 'relu')(sat_cnn)
    sat_flat = layers.Flatten()(sat_cnn)
    sat_image_embedding = layers.Dense(200, activation='relu')(sat_flat)

    # 3. Concatenate embeddings
    concat = layers.Concatenate(axis=1)([gsv_image_embedding, sat_image_embedding])

    # 4. Dense layers + classification
    full_embedding = layers.Dense(300, activation='relu')(concat)
    full_embedding = layers.Dense(100, activation='relu')(full_embedding)
    output = layers.Dense(n_classes, activation='sigmoid')(full_embedding)

    # 5. define full model and compile
    model = models.Model(inputs=[gsv_input_img, sat_input_img], outputs=output)

    adam = optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(loss=loss_fn, optimizer=adam, metrics=['accuracy', sensitivity, specificity])

    return model

def three_input_model(n_tabular_cols = 10, gsv_image_dim = (640, 640, 3), sat_image_dim = ((2100, 2100, 4)), n_classes = 1, loss_fn = 'binary_crossentropy'):

    """
    Combined CNN using satellite and Street View imagery.
    Added tabular data.
    """
    # functions from http://www.deepideas.net/unbalanced-classes-machine-learning/
    def sensitivity(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + K.epsilon())

    def specificity(y_true, y_pred):
        true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
        possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
        return true_negatives / (possible_negatives + K.epsilon())

    # 1. Google Street View (GSV) Image Input
    gsv_input_img = layers.Input(shape=gsv_image_dim, name='gsv_image_input')

    gsv_cnn = layers.Conv2D(128, (3,3), activation = 'relu')(gsv_input_img)
    gsv_cnn = layers.MaxPooling2D((2,2))(gsv_cnn)
    gsv_cnn = layers.Conv2D(64, (3,3), activation = 'relu')(gsv_cnn)
    gsv_cnn = layers.MaxPooling2D((2,2))(gsv_cnn)
    gsv_cnn = layers.Conv2D(32, (3,3), activation = 'relu')(gsv_cnn)
    gsv_flat = layers.Flatten()(gsv_cnn)
    gsv_image_embedding = layers.Dense(50, activation='relu')(gsv_flat)

    # 2. Satellite Image input
    sat_input_img = layers.Input(shape=sat_image_dim, name='aerial_image_input')

    sat_cnn = layers.Conv2D(128, (3,3), activation = 'relu')(sat_input_img)
    sat_cnn = layers.MaxPooling2D((2,2))(sat_cnn)
    sat_cnn = layers.Conv2D(64, (3,3), activation = 'relu')(sat_cnn)
    sat_cnn = layers.MaxPooling2D((2,2))(sat_cnn)
    sat_cnn = layers.Conv2D(32, (3,3), activation = 'relu')(sat_cnn)
    sat_flat = layers.Flatten()(sat_cnn)
    sat_image_embedding = layers.Dense(50, activation='relu')(sat_flat)

    # 3. Tabular Data
    tabular_input = layers.Input(shape=(n_tabular_cols,), name='tabular_input')
    tabular_layers = layers.Dense(100, activation='relu',  name='tabular_dense1')(tabular_input)
    tabular_embedding = layers.Dense(50, activation='relu')(tabular_input)

    # 4. Concatenate embeddings
    concat = layers.Concatenate(axis=1)([gsv_image_embedding, sat_image_embedding, tabular_embedding])

    # 5. Dense layers + classification
    full_embedding = layers.Dense(300, activation='relu')(concat)
    full_embedding = layers.Dense(100, activation='relu')(full_embedding)
    output = layers.Dense(n_classes, activation='sigmoid')(full_embedding)

    # 6. define full model and compile
    model = models.Model(inputs=[gsv_input_img, sat_input_img, tabular_input], outputs=output)
    adam = optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss=loss_fn, optimizer=adam, metrics=['accuracy', sensitivity, specificity])

    return model

if __name__ == "__main__":
    model = combined_cnn()
    print(model.summary())
