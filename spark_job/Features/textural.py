import numpy as np
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType, ArrayType, IntegerType
from skimage.feature import greycomatrix
from skimage.feature import greycoprops as gc


class TexturalFunctions:
    def __init__(self, df, spark):
        self.dataframe = df.selectExpr("origin","Geom","RS_Normalize(RS_GetBand(data, 1,bands)) as Band1","RS_Normalize(RS_GetBand(data, 2,bands)) as Band2","RS_Normalize(RS_GetBand(data, 3,bands)) as Band3", "RS_Normalize(RS_GetBand(data, 4,bands)) as Band4")
        self.spark = spark




    def extract_features(self):
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

        def rgbtograyscale(red, green, blue):
            a = (0.299 * np.array(red) + 0.587 * np.array(green) + 0.114 * np.array(blue)).tolist()
            b = [round(x) for x in a]
            return b

        self.spark.udf.register("GLCM_Contrast", GLCM_Contrast, DoubleType())
        self.spark.udf.register("GLCM_Dis", GLCM_Dissimilarity, DoubleType())
        self.spark.udf.register("GLCM_Hom", GLCM_Homogeneity, DoubleType())
        self.spark.udf.register("GLCM_ASM", GLCM_ASM, DoubleType())
        self.spark.udf.register("GLCM_Corr", GLCM_Correlation, DoubleType())
        self.spark.udf.register("GLCM_Energy", GLCM_Energy, DoubleType())
        self.spark.udf.register("RS_Convert", rgbtograyscale, ArrayType(IntegerType()))

        self.dataframe = self.dataframe.selectExpr("origin", "Geom", "RS_Convert(Band3, Band2, Band1) as gray_scale")
        glcm_features_df = self.dataframe.selectExpr("origin", "Geom",
                                                    "GLCM_Contrast(gray_scale) as glcm_contrast",
                                                    "GLCM_Dis(gray_scale) as glcm_dissimilarity",
                                                    "GLCM_Hom(gray_scale) as glcm_homogeneity",
                                                    "GLCM_Energy(gray_scale) as glcm_energy",
                                                    "GLCM_Corr(gray_scale) as glcm_correlation",
                                                    "GLCM_ASM(gray_scale) as glcm_ASM")

        return glcm_features_df

    def combine(self):
        df = self.extract_features()
        unlist = udf(lambda x: round(float(list(x)[0]), 3), DoubleType())

        # Iterating over columns to be scaled
        for i in ["glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity", "glcm_energy", "glcm_correlation",
                  "glcm_ASM"]:
            # VectorAssembler Transformation - Converting column to vector type
            assembler = VectorAssembler(inputCols=[i], outputCol=i + "_Vect")

            # MinMaxScaler Transformation
            scaler = MinMaxScaler(inputCol=i + "_Vect", outputCol=i + "_Scaled")

            # Pipeline of VectorAssembler and MinMaxScaler
            pipeline = Pipeline(stages=[assembler, scaler])

            # Fitting pipeline on dataframe
            df = pipeline.fit(df).transform(df).withColumn(i + "_Scaled",
                                                                                                     unlist(
                                                                                                         i + "_Scaled")).drop(
                i + "_Vect")

        glcm_features_df = df.select("origin", "Geom", "glcm_contrast_Scaled",
                                                   "glcm_dissimilarity_Scaled", "glcm_homogeneity_Scaled",
                                                   "glcm_energy_Scaled", "glcm_correlation_Scaled", "glcm_ASM_Scaled")
        glcm_features_df.show(2)









