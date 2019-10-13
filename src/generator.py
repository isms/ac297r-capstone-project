from src.raster import Raster
from os import listdir
from os.path import isfile, join
import re
import random

def data_generator(batch_size, data_dir = '../data/raster_sample/'):
	"""
	Generator function to be used during model training.

	Parameters
	----------
	batch_size : int
		Number of images to include in batch.
	data_dir : str
		Location of directory with images.

	Yields
	------
	images : np.ndarray
		Array of size (batch_size, X, Y, 3).
		batch_size = number of images
		X = dimension 1 of image (may have to pad)
		Y = dimension 2 of image (may have to pad)
		3 = number of channels (this can be tweaked in src.raster.Raster)
	"""

	raster_names = list(set([f for f in listdir(data_dir) if re.match('.*\.TIF$', f)]))


	while 1: # needed for Keras generator

		# shuffle 
		random.shuffle(raster_names) 

		# split into batches
		split_ind = list(range(0, len(raster_names), batch_size))
		batches = np.array_split(raster_names, split_ind[1:])

		for i, batch in enumerate(batches):

			print('hi')
			# rotate and resize

			# turn into np

			# pull labels


	return batches # TEMP
