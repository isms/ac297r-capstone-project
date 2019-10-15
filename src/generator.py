import matplotlib.pyplot as plt
import random
import pandas as pd
from tensorflow.keras.utils import Sequence
import numpy as np

from src.raster import Raster


class DataGenerator(Sequence):
	"""Generates data for Keras
	more info: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

	"""
	def __init__(self, batch_size = 10, image_dim = (100, 100, 5), data_dir = './data/raster_sample/', label_path='./labels/labels.csv', label_col='label'):
		"""Initialization"""
		self.batch_size = batch_size
		self.image_dim = image_dim
		self.data_dir = data_dir
		self.label_path = label_path
		self.label_col = label_col

		self.label_df = pd.read_csv(label_path, index_col = 0)
		valid_labels = [0, 1]
		self.label_df = self.label_df[self.label_df[label_col].isin(valid_labels)]
		self.label_df['filename'] = self.label_df.index

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


if __name__ == "__main__":
	dg = DataGenerator(10, (100, 100, 3))
	X, y = dg[0]
	for X, y in dg:
		for idx in range(len(y)):
			fig, ax = plt.subplots(figsize=(7, 7))
			print(y[idx])
			ax.imshow(X[idx])
			plt.show()
