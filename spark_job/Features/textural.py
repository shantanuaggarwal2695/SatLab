import numpy as np
from skimage.feature import greycomatrix
from skimage.feature import greycoprops as gc


class Textural_Features:
    def __init__(self, df, spark):
        self.dataframe = df
        self.spark = spark




    def GLCM_Contrast(self):
        def contrast(pixels):
            mi, ma = 0, 255
            ks = 5
            h, w = 32, 32
            nbit = 8
            bins = np.linspace(mi, ma + 1, nbit + 1)
            bin_image = np.digitize(pixels, bins) - 1
            new_image = np.reshape(bin_image, (32, 32))
            glcm = greycomatrix(new_image, [1], [0, np.pi / 4, np.pi / 2], levels=8, normed=True, symmetric=True)
            feature = gc(glcm, "contrast")
            return sum(feature[0].tolist()) / len(feature[0].tolist())

            self.spark.udf.register("GLCM_Contrast", contrast, DoubleType())





