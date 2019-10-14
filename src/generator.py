from src.raster import Raster
from os import listdir
from os.path import isfile, join
import re
import random
import pandas as pd

def data_generator(batch_size, data_dir = '../data/raster_sample/', 
				   label_path='../labels/labels_full.csv', label_col='driveway_label'):
	"""
	Generator function to be used during model training.

	Parameters
	----------
	batch_size : int
		Number of images to include in batch.
	data_dir : str
		Location of directory with images.
	label_path : str
		Location of csv with labels.
	label_col : str
		Name of column with label.

	Yields
	------
	images : np.ndarray
		Array of size (batch_size, X, Y, 3).
			batch_size = number of images
			X = dimension 1 of image (may have to pad)
			Y = dimension 2 of image (may have to pad)
			3 = number of channels (this can be tweaked in src.raster.Raster)
	image_labels : array-like
		Array of size batch_size. 
	"""

	# TEMP
	images = 1
	image_labels = 1

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


		yield (images, image_labels) 


def simple_data_generator(data_dir = '../data/raster_sample/', 
						  label_path='../labels/labels_full.csv', 
						  label_col='driveway_label'):
	"""
	Simple Generator function to be used during model training. This 
	function has no batch_size argument.

	Parameters
	----------
	data_dir : str
		Location of directory with images.
	label_path : str
		Location of csv with labels.
	label_col : str
	Name of column with label.

	Yields
	------
	images : np.ndarray
		Array of size (n, X, Y, 3).
			label_path.shape[0] = number of images
			X = dimension 1 of image (may have to pad)
			Y = dimension 2 of image (may have to pad)
			3 = number of channels (this can be tweaked in src.raster.Raster)
	image_labels : array-like
		Array of size label_path.shape[0]. 
	"""

	# TEMP
	images = 1

	# set up
	#raster_names = list(set([f for f in listdir(data_dir) if re.match('.*\.TIF$', f)]))
	labels_df = pd.read_csv(label_path)
	raster_names = labels_df.iloc[:,0].to_list()
	print(raster_names)


	#while 1: # needed for Keras generator

	# read in .tif files
	all_rasters = []
	label_row_inds = []
	for i, f in enumerate(raster_names):
		try:
			img = Raster(data_dir, f)
			all_rasters.append(img.arr)
			label_row_inds.append(i)
		except:
			pass

	# rotate and resize

	# turn into np

	# pull labels
	image_labels = labels_df[labels_df[label_col].isin(raster_names)][label_col]

	return (all_rasters, image_labels)

