import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from IPython import display

class Raster:
    def __init__(self, data_dir, image_name):
        # read raster
        self.parcel_name = image_name
        self.src = rasterio.open(data_dir + image_name)

        # extract RGB and transpose into (row, col, band)
        self.arr = self.src.read([1,2,3]).transpose(1,2,0)

    def show(self):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(self.arr)
        plt.show()
