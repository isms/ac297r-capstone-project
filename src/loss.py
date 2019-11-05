import keras.backend as K
import tensorflow as tf

def smooth_labels(labels_str):
	'''
	Converts concatenated labels into smoothed floating point labels.

	Parameters
	----------
	labels_str : array-like
		Labels. String in the format 'aerial_X_gsv_Y' where
			X : satellite label
			Y : Google Streetview label

	Returns
	-------
	smoothed_labels : array-like
		Smoothed labels.
	'''


	# def _parse_data(line):
    # line_split = tf.string_split([line], '\t')
	#
    # raw_text = tf.py_func(
    #     lambda x: x.strip().lower(), line_split.values[0], tf.string)
	#
    # label = tf.string_to_number(line_split.values[1], out_type=tf.int32)
	#
    # return {"raw_text": raw_text, "label": label}
	#



	# split string column for label
	split_label = tf.string_split(labels_str,'_')
	aerial_label = split_label[1]
	gsv_label = split_label[3]

	# smoothing criteria

	## both labels agree
	if aerial_label == '1' and gsv_label == '1':
		smoothed_label = 1.
	elif aerial_label == '0' and gsv_label == '0':
		smoothed_label = 0.

	## labels disagree
	elif aerial_label == '1' and gsv_label == '0':
		smoothed_label = 0.9
	elif aerial_label == '0' and gsv_label == '1':
		smoothed_label = 0.9

	## both labels uncertain
	elif aerial_label == '2' and gsv_label == '2':
		smoothed_label = 0.5

	## one label certain, one label unsure
	elif aerial_label == '2' and gsv_label != '2':
		if gsv_label == '1':
			smoothed_label = 0.9
		elif gsv_label == '0':
			smoothed_label = 0.1
	elif aerial_label != '2' and gsv_label == '2':
		if aerial_label == '1':
			smoothed_label = 0.9
		elif aerial_label == '0':
			smoothed_label = 0.1

	smoothed_labels.append(smoothed_label)

	return smoothed_label


def smoothed_binary_crossentropy(y_true, y_pred):
	'''
	Binary crossentropy loss function to work with pre-smoothed labels.
	Adapted from Keras binary crossentropy source code:
	http://github.com/keras-team/keras/blob/master/keras/losses.py

	Parameters
	----------
	y_true : array-like
		True labels. Should be of form 'aerial_X_gsv_Y' where
			X : satellite label
			Y : Google Streetview label
	y_pred : array-like
		Predicted labels. Should be floats in [0, 1].

	Returns
	-------
	bce : float
		Binary cross-entropy.
	'''

	smoothed_y_true = tf.map_fn(smooth_labels, y_true)

	y_pred = K.constant(y_pred) #if not K.is_tensor(y_pred) else y_pred
	smoothed_y_true = K.cast(smoothed_y_true, y_pred.dtype)

	bce = K.mean(K.binary_crossentropy(smoothed_y_true, y_pred), axis=-1)

	return bce


if __name__ == "__main__":

	import pandas as pd
	import numpy as np

	labels = pd.read_csv('../data/training_labels.csv')
	labels['full_label'] = 'aerial_' + labels['AERIAL_Driveway'].astype(int).astype(str) + \
						   '_gsv_' + labels['GSV_Driveway'].astype(int).astype(str)

	y_true = labels['full_label']
	y_pred = np.random.uniform(size=len(y_true)) # fake predictions

	bce = smoothed_binary_crossentropy(y_true, y_pred)

	print(bce)
