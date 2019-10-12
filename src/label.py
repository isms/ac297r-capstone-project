from src.raster import Raster
import re
from os import listdir
from os.path import isfile, join
from IPython.display import clear_output
import pandas as pd

data_dir = '../data/raster_sample/'
label_path = '../labels/labels_full.csv' #'../labels/labels.csv'

def load_labels(label_path):
    df = pd.read_csv(label_path, index_col=0)
    return set(df.index)

def get_label(f):
    img = Raster(data_dir, f)
    img.show()

    driveway_label = input("Enter 1 for driveway, 0 for no driveway, and 2 for don't know:")
    capacity_label = input("Enter integer estimate for how many cars would fit in driveway, -1 if no driveway:")
    garage_label = input("Enter 1 for garage, 0 for no garage, and 2 for don't know:")
    clear_output()

    return [driveway_label, capacity_label, garage_label]

def save_label(r, label, label_path):
    driveway, capacity, garage = label
    with open(label_path,'a') as label_data:
        label_data.write(f"{r}, {driveway}, {capacity}, {garage}\n")

def label():
    raster_names = set([f for f in listdir(data_dir) if re.match('.*\.TIF$', f)])

    already_labelled = load_labels(label_path)
    to_be_labelled = raster_names - already_labelled

    for r in to_be_labelled:
        label = get_label(r)
        save_label(r, label, label_path)
