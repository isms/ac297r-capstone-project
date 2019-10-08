import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt

class Raster:
    def __init__(self, data_dir, image_name):
        # read raster
        self.parcel_name = image
        self.src = rasterio.open(data_dir + image)
        # extract RGB and transpose into (row, col, band)
        self.arr = self.src.read([1,2,3]).transpose(1,2,0)

    def __str__(self):
        plt.imshow(self.arr)
        return self.parcel_name
