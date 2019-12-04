import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.applications.inception_v3 import InceptionV3


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

def street_view_cnn(pretrained_model, image_dim, optimizer, loss, n_classes=3, activation = 'softmax', metrics=['accuracy']):
    '''
    Parameters:
    - pretrained_model: pretrained model with trainable layers previously defined, make sure top layer is not included
    - image_dim: input image dimensions
    - optimizer: predefined optimizer
    - loss: predefined loss
    
    Returns:
    Compiled model
    '''
    
    # 1. Prepare imput image
    sv_input_img = keras.layers.Input(shape=image_dim, name='streetview_image_input')
    
    # 2. Pass input image through pretrained model - note that this model is set to trainable/non0tr
    pre_trained_embedding = pretrained_model(sv_input_img)
    
    # 3. Flatten image embedding
    flat_pre_trained_embedding = keras.layers.Flatten()(pre_trained_embedding)
    
    # 4. Run through some dense layers with batch normalization and dropout
    full_embedding = keras.layers.Dense(100,
                kernel_regularizer=keras.regularizers.l2(0.01))(flat_pre_trained_embedding)
    full_embedding = keras.layers.BatchNormalization()(full_embedding)
    full_embedding = keras.layers.LeakyReLU(alpha=0.2)(full_embedding)

    full_embedding = keras.layers.Dropout(rate=0.3)(full_embedding)

#     full_embedding = keras.layers.Dense(100, 
#                 kernel_regularizer=keras.regularizers.l2(0.01))(full_embedding)
#     full_embedding = keras.layers.BatchNormalization()(full_embedding)
#     full_embedding = keras.layers.LeakyReLU(alpha=0.3)(full_embedding)

#     full_embedding = keras.layers.Dropout(rate=0.2)(full_embedding)

    full_embedding = keras.layers.Dense(50,
                kernel_regularizer=keras.regularizers.l2(0.01))(full_embedding)
    full_embedding = keras.layers.BatchNormalization()(full_embedding)
    full_embedding = keras.layers.LeakyReLU(alpha=0.3)(full_embedding)

    full_embedding = keras.layers.Dropout(rate=0.2)(full_embedding)
    
    # 5. Define output
    output = keras.layers.Dense(n_classes, activation=activation)(full_embedding)
    
    # 6. Define Model 
    model = keras.models.Model(inputs=sv_input_img, outputs=output)
    
    # 7. Compine Model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    
    return model
def aerial_cnn(pretrained_model, image_dim, optimizer, loss, n_classes=3, activation = 'softmax', metrics=['accuracy']):
    '''
    Parameters:
    - pretrained_model: pretrained model with trainable layers previously defined, make sure top layer is not included
    - image_dim: input image dimensions
    - optimizer: predefined optimizer
    - loss: predefined loss
    
    Returns:
    Compiled model
    '''
    
    # 1. Prepare imput image
    aerial_input_img = keras.layers.Input(shape=image_dim, name='satellite_image_input')
    
    # 2. Pass input image through pretrained model - note that this model is set to trainable/non0tr
    pre_trained_embedding = pretrained_model(aerial_input_img)
    
    # 3. Flatten image embedding
    flat_pre_trained_embedding = keras.layers.Flatten()(pre_trained_embedding)
    
    # 4. Run through some dense layers with batch normalization and dropout
    full_embedding = keras.layers.Dense(100,
                kernel_regularizer=keras.regularizers.l2(0.01))(flat_pre_trained_embedding)
    full_embedding = keras.layers.BatchNormalization()(full_embedding)
    full_embedding = keras.layers.LeakyReLU(alpha=0.2)(full_embedding)

#     full_embedding = keras.layers.Dropout(rate=0.3)(full_embedding)

#     full_embedding = keras.layers.Dense(100, 
#                 kernel_regularizer=keras.regularizers.l2(0.01))(full_embedding)
#     full_embedding = keras.layers.BatchNormalization()(full_embedding)
#     full_embedding = keras.layers.LeakyReLU(alpha=0.3)(full_embedding)

    full_embedding = keras.layers.Dropout(rate=0.3)(full_embedding)

    full_embedding = keras.layers.Dense(50,
                kernel_regularizer=keras.regularizers.l2(0.01))(full_embedding)
    full_embedding = keras.layers.BatchNormalization()(full_embedding)
    full_embedding = keras.layers.LeakyReLU(alpha=0.3)(full_embedding)

    full_embedding = keras.layers.Dropout(rate=0.2)(full_embedding)
    
    # 5. Define output
    output = keras.layers.Dense(n_classes, activation=activation)(full_embedding)
    
    # 6. Define Model 
    model = keras.models.Model(inputs=aerial_input_img, outputs=output)
    
    # 7. Compine Model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    
    return model

def combined_cnn(gsv_image_dim = (640, 640, 3), sat_image_dim = ((2100, 2100, 4)), n_classes = 2, loss_fn = 'binary_crossentropy', final_layer_activation='sigmoid', learning_rate = 1e-3):

    """
    Combined CNN using satellite and Street View imagery.
    """

    # 1. Google Street View (GSV) Image Input
    gsv_input_img = layers.Input(shape=gsv_image_dim, name='gsv_image_input')

    gsv_cnn = layers.Conv2D(256, (3,3), activation = 'relu')(gsv_input_img)
    gsv_cnn = layers.MaxPooling2D((2,2))(gsv_cnn)
    gsv_cnn = layers.Conv2D(128, (3,3), activation = 'relu')(gsv_cnn)
    gsv_cnn = layers.MaxPooling2D((2,2))(gsv_cnn)
    gsv_cnn = layers.Conv2D(128, (3,3), activation = 'relu')(gsv_cnn)
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
    sat_cnn = layers.Conv2D(128, (3,3), activation = 'relu')(sat_cnn)
    sat_cnn = layers.MaxPooling2D((2,2))(sat_cnn)
    sat_cnn = layers.Conv2D(64, (3,3), activation = 'relu')(sat_cnn)
    sat_flat = layers.Flatten()(sat_cnn)
    sat_image_embedding = layers.Dense(200, activation='relu')(sat_flat)

    # 3. Concatenate embeddings
    concat = layers.Concatenate(axis=1)([gsv_image_embedding, sat_image_embedding])

    # 4. Dense layers + classification
    full_embedding = layers.BatchNormalization()(concat)
    full_embedding = layers.Dense(100, activation='relu')(full_embedding)
    full_embedding = layers.Dense(50, activation='relu')(full_embedding)
    output = layers.Dense(n_classes, activation=final_layer_activation)(full_embedding)

    # 5. define full model and compile
    model = models.Model(inputs=[gsv_input_img, sat_input_img], outputs=output)

    adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(loss=loss_fn, optimizer=adam, metrics=['accuracy'])

    return model

def three_input_model(n_tabular_cols = 10, gsv_image_dim = (640, 640, 3), sat_image_dim = ((2100, 2100, 4)), n_classes = 1, loss_fn = 'binary_crossentropy',final_layer_activation='softmax'):

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

    gsv_cnn = layers.Conv2D(256, (3,3))(gsv_input_img)
    gsv_cnn = layers.BatchNormalization()(gsv_cnn)
    gsv_cnn = layers.Activation("relu")(gsv_cnn)
    gsv_cnn = layers.Dropout(0.2)(gsv_cnn)
    
    gsv_cnn = layers.MaxPooling2D((2,2))(gsv_cnn)
    
    gsv_cnn = layers.Conv2D(128, (3,3))(gsv_cnn)
    gsv_cnn = layers.BatchNormalization()(gsv_cnn)
    gsv_cnn = layers.Activation("relu")(gsv_cnn)
    gsv_cnn = layers.Dropout(0.2)(gsv_cnn)

    gsv_cnn = layers.MaxPooling2D((2,2))(gsv_cnn)
    
    gsv_cnn = layers.Conv2D(64, (3,3))(gsv_cnn)
    gsv_cnn = layers.BatchNormalization()(gsv_cnn)
    gsv_cnn = layers.Activation("relu")(gsv_cnn)
    gsv_cnn = layers.Dropout(0.2)(gsv_cnn)

    gsv_flat = layers.Flatten()(gsv_cnn)
    gsv_image_embedding = layers.Dense(100, activation='relu')(gsv_flat)
    
    # 2. Satellite Image input
    sat_input_img = layers.Input(shape=sat_image_dim, name='aerial_image_input')
    sat_cnn = layers.Dropout(0.2)(sat_input_img)
    
    sat_cnn = layers.Conv2D(256, (3,3))(sat_cnn)
    sat_cnn = layers.BatchNormalization()(sat_cnn)
    sat_cnn = layers.Activation("relu")(sat_cnn)
    sat_cnn = layers.Dropout(0.2)(sat_cnn)
    
    sat_cnn = layers.MaxPooling2D((2,2))(sat_cnn)
    
    sat_cnn = layers.Conv2D(128, (3,3))(sat_cnn)
    sat_cnn = layers.BatchNormalization()(sat_cnn)
    sat_cnn = layers.Activation("relu")(sat_cnn)
    sat_cnn = layers.Dropout(0.2)(sat_cnn)

    sat_cnn = layers.MaxPooling2D((2,2))(sat_cnn)
    
    sat_cnn = layers.Conv2D(64, (3,3))(sat_cnn)
    sat_cnn = layers.BatchNormalization()(sat_cnn)
    sat_cnn = layers.Activation("relu")(sat_cnn)
    sat_cnn = layers.Dropout(0.2)(sat_cnn)

    sat_flat = layers.Flatten()(sat_cnn)
    sat_image_embedding = layers.Dense(100, activation='relu')(sat_flat)

    # 3. Tabular Data
    tabular_input = layers.Input(shape=(n_tabular_cols,), name='tabular_input')
    tabular_layers = layers.Dense(200, activation='relu',  name='tabular_dense1')(tabular_input)
    tabular_embedding = layers.Dense(100, activation='relu')(tabular_input)

    # 4. Concatenate embeddings
    concat = layers.Concatenate(axis=1)([gsv_image_embedding, sat_image_embedding, tabular_embedding])

    # 5. Dense layers + classification
    full_embedding = layers.Dense(150, activation='relu')(concat)
    full_embedding = layers.Dense(50, activation='relu')(full_embedding)
    output = layers.Dense(n_classes, activation = final_layer_activation)(full_embedding)

    # 6. define full model and compile
    model = models.Model(inputs=[gsv_input_img, sat_input_img, tabular_input], outputs=output)
    adam = optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss=loss_fn, optimizer=adam, metrics=['accuracy', sensitivity, specificity])

    return model

if __name__ == "__main__":
    model = combined_cnn()
    print(model.summary())
