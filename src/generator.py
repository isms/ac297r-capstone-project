import matplotlib.pyplot as plt
import random
import pandas as pd
from tensorflow.keras.utils import Sequence
import numpy as np
from src.raster import Raster

from keras_preprocessing.image import ImageDataGenerator

### New generator for model with two inputs
def gen_flow_for_two_inputs(sample, aerial_dir = '../data/training/aerial_images/', gsv_dir ='../data/training/street_view_images/', batch_size = 32, gsv_image_dim = (640, 640, 3), aer_image_dim = ((2100, 2100, 4))):

	aerial_gen = ImageDataGenerator(horizontal_flip = True, vertical_flip = True, width_shift_range = 0.1,height_shift_range = 0.1, zoom_range = 0.1, rotation_range = 40)
	gsv_gen = ImageDataGenerator()

	# generator for aerial images
	aerial_gen = aerial_gen.flow_from_dataframe(sample, directory = aerial_dir, x_col= 'aerial_filename', y_col='temp_label', target_size=(aer_image_dim[0], aer_image_dim[1]), color_mode='rgba', class_mode='categorical', batch_size=batch_size, shuffle=True, seed=100)

	# generator for gsv images
	gsv_gen = aerial_gen.flow_from_dataframe(sample, directory = gsv_dir, x_col= 'aerial_filename', y_col='temp_label', target_size=(gsv_image_dim[0], gsv_image_dim[1]), color_mode='rgb',class_mode='categorical',batch_size=batch_size,shuffle=True,seed=100)

	# put both together
	while True:
		X1_i = aerial_gen.next()
		X2_i = gsv_gen.next()
		#Assert arrays are equal - this was for peace of mind, but slows down training
		#np.testing.assert_array_equal(X1i[0],X2i[0])
		yield [X1_i[0], X2_i[1]], X1_i[1]



class DataGenerator(Sequence):
	"""Generates data for Keras
	more info: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

	"""
	def __init__(self, label_df, batch_size = 10, image_dim = (100, 100, 5), data_dir = './data/raster_sample/', label_col='label'):
		"""Initialization"""
		self.batch_size = batch_size
		self.image_dim = image_dim
		self.data_dir = data_dir
		self.label_col = label_col
		self.label_df = label_df


	def __len__(self):
		"""Denotes the number of batches per epoch"""
		return int(np.floor(len(self.label_df) / self.batch_size))

	def __getitem__(self, index):
		"""Generate one batch of data"""
		# get rows of label_df corresponding to batch
		batch_df = self.label_df.iloc[index*self.batch_size:(index+1)*self.batch_size]

		# get labels
		y = batch_df[self.label_col].values

		X = self.get_X(batch_df)

		return X, y

	def on_epoch_end(self):
		"""Updates indexes after each epoch"""
		self.label_df = self.label_df.sample(frac=1)


	def get_X(self, batch_df):
		"""Loads images with proper dimensions"""
		X_dim = tuple([len(batch_df)]) + self.image_dim
		X = np.zeros(X_dim)
		for idx, file in enumerate(batch_df.filename):
			r = Raster(self.data_dir, file)
			X[idx] = r.clean(self.image_dim)
		return X

def load_images(label_df, label_col, image_dim, data_dir = './data/raster_sample/'):
	y = label_df[label_col].values

	X_dim = tuple([len(label_df)]) + image_dim
	X = np.zeros(X_dim)
	for idx, file in enumerate(label_df.filename):
		r = Raster(data_dir, file)
		X[idx] = r.clean(image_dim)
	return X, y

if __name__ == "__main__":
	dg = DataGenerator(10, (100, 100, 3))
	X, y = dg[0]
	for X, y in dg:
		for idx in range(len(y)):
			fig, ax = plt.subplots(figsize=(7, 7))
			print(y[idx])
			ax.imshow(X[idx])
			plt.show()
