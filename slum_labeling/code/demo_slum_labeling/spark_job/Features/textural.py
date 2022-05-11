from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, IntegerType, DoubleType
import numpy as np
from skimage.feature import *
from skimage.feature import greycoprops as gc
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf

class Textural:
    def __init__(self, train,spark):
        self.train = train
        # self.final_train = self.train.selectExpr("image.origin as origin", "ST_GeomFromWkt(image.wkt) as Geom",
        #                                     "image.height as height", "image.width as width", "image.data as data",
        #                                     "image.nBands as bands")
        self.final_train = self.train.selectExpr("origin", "Geom", "RS_Normalize(RS_GetBand(data, 1,bands)) as Band1",
                                             "RS_Normalize(RS_GetBand(data, 2,bands)) as Band2",
                                             "RS_Normalize(RS_GetBand(data, 3,bands)) as Band3",
                                             "RS_Normalize(RS_GetBand(data, 4,bands)) as Band4")
        self.spark = spark

    def gray_scale(self):
        def rgbtograyscale(red, green, blue):
            a = (0.299 * np.array(red) + 0.587 * np.array(green) + 0.114 * np.array(blue)).tolist()
            b = [round(x) for x in a]
            return b

        rgb_grayscale = udf(rgbtograyscale, ArrayType(IntegerType()))
        self.spark.udf.register("RS_Convert", rgb_grayscale)

        gray_scale_dF = self.final_train.selectExpr("origin", "Geom", "RS_Convert(Band3, Band2, Band1) as gray_scale")
        return gray_scale_dF

    def extract_features(self):
        gray_scale= self.gray_scale()

        def GLCM_Contrast(pixels):
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

        def GLCM_Dissimilarity(pixels):
            mi, ma = 0, 255
            ks = 5
            h, w = 32, 32
            nbit = 8
            bins = np.linspace(mi, ma + 1, nbit + 1)
            bin_image = np.digitize(pixels, bins) - 1
            new_image = np.reshape(bin_image, (32, 32))
            glcm = greycomatrix(new_image, [1], [0, np.pi / 4, np.pi / 2], levels=8, normed=True, symmetric=True)
            feature = gc(glcm, "dissimilarity")
            return sum(feature[0].tolist()) / len(feature[0].tolist())

        def GLCM_Homogeneity(pixels):
            mi, ma = 0, 255
            ks = 5
            h, w = 32, 32
            nbit = 8
            bins = np.linspace(mi, ma + 1, nbit + 1)
            bin_image = np.digitize(pixels, bins) - 1
            new_image = np.reshape(bin_image, (32, 32))
            glcm = greycomatrix(new_image, [1], [0, np.pi / 4, np.pi / 2], levels=8, normed=True, symmetric=True)
            feature = gc(glcm, "homogeneity")
            return sum(feature[0].tolist()) / len(feature[0].tolist())

        def GLCM_Energy(pixels):
            mi, ma = 0, 255
            ks = 5
            h, w = 32, 32
            nbit = 8
            bins = np.linspace(mi, ma + 1, nbit + 1)
            bin_image = np.digitize(pixels, bins) - 1
            new_image = np.reshape(bin_image, (32, 32))
            glcm = greycomatrix(new_image, [1], [0, np.pi / 4, np.pi / 2], levels=8, normed=True, symmetric=True)
            feature = gc(glcm, "energy")
            return sum(feature[0].tolist()) / len(feature[0].tolist())

        def GLCM_Correlation(pixels):
            mi, ma = 0, 255
            ks = 5
            h, w = 32, 32
            nbit = 8
            bins = np.linspace(mi, ma + 1, nbit + 1)
            bin_image = np.digitize(pixels, bins) - 1
            new_image = np.reshape(bin_image, (32, 32))
            glcm = greycomatrix(new_image, [1], [0, np.pi / 4, np.pi / 2], levels=8, normed=True, symmetric=True)
            feature = gc(glcm, "correlation")
            return sum(feature[0].tolist()) / len(feature[0].tolist())

        def GLCM_ASM(pixels):
            mi, ma = 0, 255
            ks = 5
            h, w = 32, 32
            nbit = 8
            bins = np.linspace(mi, ma + 1, nbit + 1)
            bin_image = np.digitize(pixels, bins) - 1
            new_image = np.reshape(bin_image, (32, 32))
            glcm = greycomatrix(new_image, [1], [np.pi / 4, np.pi / 2], levels=8, normed=True, symmetric=True)
            feature = gc(glcm, "ASM")
            return sum(feature[0].tolist()) / len(feature[0].tolist())

        self.spark.udf.register("GLCM_Contrast", GLCM_Contrast, DoubleType())
        self.spark.udf.register("GLCM_Dis", GLCM_Dissimilarity, DoubleType())
        self.spark.udf.register("GLCM_Hom", GLCM_Homogeneity, DoubleType())
        self.spark.udf.register("GLCM_ASM", GLCM_ASM, DoubleType())
        self.spark.udf.register("GLCM_Corr", GLCM_Correlation, DoubleType())
        self.spark.udf.register("GLCM_Energy", GLCM_Energy, DoubleType())

        glcm_features_df = gray_scale.selectExpr("origin", "ST_Centroid(Geom) as Geom",
                                                    "GLCM_Contrast(gray_scale) as glcm_contrast",
                                                    "GLCM_Dis(gray_scale) as glcm_dissimilarity",
                                                    "GLCM_Hom(gray_scale) as glcm_homogeneity",
                                                    "GLCM_Energy(gray_scale) as glcm_energy",
                                                    "GLCM_Corr(gray_scale) as glcm_correlation",
                                                    "GLCM_ASM(gray_scale) as glcm_ASM")

        # unlist = udf(lambda x: round(float(list(x)[0]), 3), DoubleType())
        #
        # # Iterating over columns to be scaled
        # for i in ["glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity", "glcm_energy", "glcm_correlation",
        #           "glcm_ASM"]:
        #     # VectorAssembler Transformation - Converting column to vector type
        #     assembler = VectorAssembler(inputCols=[i], outputCol=i + "_Vect")
        #
        #     # MinMaxScaler Transformation
        #     scaler = MinMaxScaler(inputCol=i + "_Vect", outputCol=i + "_Scaled")
        #
        #     # Pipeline of VectorAssembler and MinMaxScaler
        #     pipeline = Pipeline(stages=[assembler, scaler])
        #
        #     # Fitting pipeline on dataframe
        #     glcm_features_df = pipeline.fit(glcm_features_df).transform(glcm_features_df).withColumn(i + "_Scaled",
        #                                                                                              unlist(
        #                                                                                                  i + "_Scaled")).drop(
        #         i + "_Vect")
        #
        # glcm_features_df = glcm_features_df.select("origin", "Geom", "glcm_contrast_Scaled",
        #                                            "glcm_dissimilarity_Scaled", "glcm_homogeneity_Scaled",
        #                                            "glcm_energy_Scaled", "glcm_correlation_Scaled", "glcm_ASM_Scaled")
        glcm_features_df.show()
        return glcm_features_df






