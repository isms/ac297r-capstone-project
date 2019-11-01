import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from IPython import display
from skimage.transform import resize

class Raster:
    def __init__(self, data_dir, image_name):
        # read raster
        self.parcel_name = image_name
        self.src = rasterio.open(data_dir + image_name)

        # extract RGB and transpose into (row, col, channel)
        self.arr = (self.src
            .read([1,2,3])
            .transpose(1,2,0)
            .astype(float)
        )
        self.arr = self.arr / 256

    def show(self):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(self.arr)
        plt.show()

    def savefig(self, *args, **kwargs):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(self.arr)
        plt.savefig(*args, **kwargs)
        plt.show()

    def clean(self, image_dim):
        height, width, channels = image_dim
        image = (self.src
            .read(range(1, channels + 1))
            .transpose(1,2,0)
            .astype(float)
        )
        image = image / 256
        image = resize(image, (height, width))
        return image

if __name__ == "__main__":
    data_dir = './data/raster_sample/'
    name = "parcel998.TIF"
    r = Raster(data_dir, name)
    img = r.clean((64,64,5))
    print(img.shape)
