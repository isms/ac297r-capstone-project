import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from src.models import baseline_cnn
from src.generator import DataGenerator, load_images

def train_baseline(image_dim, epochs, batch_size, label_path='./labels/labels.csv', label_col='label'):
    # load labelled data
    label_df = pd.read_csv(label_path)

    # load labelled data
    valid_labels = [0, 1]
    # TEMP FIX
    bad_parcels = ['parcel9999.TIF']
    label_df = label_df[(label_df[label_col].isin(valid_labels)) & (~label_df[filename].isin(bad_parcels))]

    #balance training data
    label_df_1 = label_df[label_df[label_col] == 1]
    label_df_0 = label_df[label_df[label_col] == 0]
    if len(label_df_1) > len(label_df_0):
        label_df_0 = label_df_0.sample(n=len(label_df_1), replace = True)
    else:
        label_df_1 = label_df_1.sample(n=len(label_df_0), replace = True)
    label_df = pd.concat([label_df_1, label_df_0]).sample(frac = 1)

    # split into train and validation sets
    train, test = train_test_split(label_df, test_size=0.2, random_state=42)

    # initialize generators
    training_generator = DataGenerator(train, batch_size = batch_size, image_dim = image_dim, 
                                       data_dir = data_dir, label_col = label_col)
    validation_generator = DataGenerator(test, batch_size = len(test), image_dim = image_dim, 
                                         data_dir = data_dir, label_col = label_col)

    # initialize model
    model = baseline_cnn(image_dim)

    # Train model on dataset
    model.fit_generator(generator=training_generator,
                        epochs = epochs,
                        validation_data=validation_generator
    )
    return model

def train_data_augmentation(image_dim, epochs, batch_size, label_path='./labels/labels.csv', 
                            label_col='label', data_dir='./data/raster_sample/'):
    # load labelled data
    label_df = pd.read_csv(label_path)

    # load labelled data
    valid_labels = [0, 1]
       # TEMP FIX
    bad_parcels = ['parcel9999.TIF']
    label_df = label_df[(label_df[label_col].isin(valid_labels)) & (~label_df.filename.isin(bad_parcels))]

    # split data
    train, test = train_test_split(label_df, test_size=0.2, random_state=42)
    #print(test)

    # initialize validation data generator
    validation_generator = DataGenerator(test, batch_size = len(test), image_dim = image_dim, 
                                         data_dir = data_dir, label_col = label_col)

    #balance training data
    train_1 = train[train[label_col] == 1]
    train_0 = train[train[label_col] == 0]
    if len(train_1) > len(train_0):
        train_0 = train_0.sample(n=len(train_1), replace = True)
    else:
        train_1 = train_1.sample(n=len(train_0), replace = True)
    train = pd.concat([train_1, train_0]).sample(frac = 1)

    X_train, y_train = load_images(train, label_col, image_dim, data_dir)

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)
    # initialize model
    model = baseline_cnn(image_dim)

    # fits the model on batches with real-time data augmentation:
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                        validation_data = validation_generator,
                        steps_per_epoch=len(X_train) / batch_size, epochs=epochs)

    return model, datagen


if __name__ == "__main__":
    image_dim = (100, 100, 5)
    epochs = 100
    batch_size = 32
    model = train_data_augmentation(image_dim, epochs, batch_size)
