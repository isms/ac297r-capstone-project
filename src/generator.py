import matplotlib.pyplot as plt
import random
import pandas as pd
from tensorflow.keras.utils import Sequence
import numpy as np
# from src.raster import Raster
from tensorflow.keras.preprocessing.image import ImageDataGenerator

### New generator for model with two inputs
def generator_two_inputs(sample, aerial_dir = '../data/training/aerial_images/', gsv_dir ='../data/training/sv_images/', batch_size = 32, aer_image_dim = (2100, 2100, 4), gsv_image_dim = (640, 640, 3) , y_column = 'final_label'):

	# generator for gsv images
	gsv_gen_obj = ImageDataGenerator()
	gsv_gen = gsv_gen_obj.flow_from_dataframe(sample, directory = '../data/training/sv_images/', x_col= 'gsv_filename', y_col=y_column, target_size=(gsv_image_dim[0], gsv_image_dim[1]), color_mode='rgb',class_mode='binary',batch_size=batch_size,shuffle=True,seed=100)

	# generator for aerial images
	aerial_gen_obj = ImageDataGenerator(horizontal_flip = True, vertical_flip = True, width_shift_range = 0.1,height_shift_range = 0.1, zoom_range = 0.1, rotation_range = 40)
	aerial_gen = aerial_gen_obj.flow_from_dataframe(sample, directory = aerial_dir, x_col= 'aerial_filename', y_col= y_column, target_size=(aer_image_dim[0], aer_image_dim[1]), color_mode='rgba', class_mode='binary', batch_size=batch_size, shuffle=True, seed=100)

	# put both together
	while True:
		gsv_i = gsv_gen.next()
		aerial_i = aerial_gen.next()
		# print(X1_i[1])
		# print(X1_i[1].shape)
		#Assert arrays are equal - this was for peace of mind, but slows down training
		#np.testing.assert_array_equal(X1i[0],X2i[0])
		yield [gsv_i[0], aerial_i[0]], gsv_i[1]

### New generator for model with two image inputs and tabular inputs
def generator_three_inputs(sample, tabular_data, tabular_predictor_cols, aerial_dir = '../data/training/aerial_images/', gsv_dir ='../data/training/sv_images/', batch_size = 32, aer_image_dim = (2100, 2100, 4), gsv_image_dim = (640, 640, 3) , y_column = 'final_label'):

	# generator for gsv images
	gsv_gen_obj = ImageDataGenerator()
	gsv_gen = gsv_gen_obj.flow_from_dataframe(sample, directory = '../data/training/sv_images/', x_col= 'gsv_filename', y_col='MBL', target_size=(gsv_image_dim[0], gsv_image_dim[1]), color_mode='rgb',class_mode='raw',batch_size=batch_size,shuffle=True,seed=100)

	# generator for aerial images
	aerial_gen_obj = ImageDataGenerator(horizontal_flip = True, vertical_flip = True, width_shift_range = 0.1,height_shift_range = 0.1, zoom_range = 0.1, rotation_range = 40)
	aerial_gen = aerial_gen_obj.flow_from_dataframe(sample, directory = aerial_dir, x_col= 'aerial_filename', y_col= y_column, target_size=(aer_image_dim[0], aer_image_dim[1]), color_mode='rgba', class_mode='binary', batch_size=batch_size, shuffle=True, seed=100)

	# put both together
	while True:
		gsv_i = gsv_gen.next()
		aerial_i = aerial_gen.next()
		tabular_data_i = (tabular_data.loc[tabular_data.MBL.isin(gsv_i[1]), tabular_predictor_cols]).values
		# print(tabular_data_i.shape)
		# print(aerial_i[1])

		yield [gsv_i[0], aerial_i[0], tabular_data_i], aerial_i[1]


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
