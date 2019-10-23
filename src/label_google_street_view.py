import re
from os import listdir
from os.path import isfile, join
from IPython.display import Image, display, clear_output
import pandas as pd

data_dir = '../data/street_view/'
label_path = '../labels/labels_google_street_view.csv'

def load_labels(label_path):
    df = pd.read_csv(label_path)
    return set(df.filename)

def get_label(f):
    img = Image(data_dir + f)
    display(img)

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
    images_names = set([f for f in listdir(data_dir) if re.match('.*\.jpg$', f)])

    already_labelled = load_labels(label_path)
    to_be_labelled = images_names - already_labelled

    for r in to_be_labelled:
        label = get_label(r)
        save_label(r, label, label_path)
