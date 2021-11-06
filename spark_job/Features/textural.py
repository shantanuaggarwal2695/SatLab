import numpy as np
from skimage.feature import greycomatrix
from skimage.feature import greycoprops as gc


class TexturalFunctions:
    def __init__(self, df, spark):
        self.dataframe = df.selectExpr("origin","Geom","RS_Normalize(RS_GetBand(data, 1,bands)) as Band1","RS_Normalize(RS_GetBand(data, 2,bands)) as Band2","RS_Normalize(RS_GetBand(data, 3,bands)) as Band3", "RS_Normalize(RS_GetBand(data, 4,bands)) as Band4")
        self.spark = spark
        self.dataframe = self.dataframe.selectExpr("origin", "Geom", "RS_Convert(Band3, Band2, Band1) as gray_scale")


    def extract_features(self):
        glcm_features_df = self.dataframe.selectExpr("origin", "Geom",
                                                    "GLCM_Contrast(gray_scale) as glcm_contrast",
                                                    "GLCM_Dis(gray_scale) as glcm_dissimilarity",
                                                    "GLCM_Hom(gray_scale) as glcm_homogeneity",
                                                    "GLCM_Energy(gray_scale) as glcm_energy",
                                                    "GLCM_Corr(gray_scale) as glcm_correlation",
                                                    "GLCM_ASM(gray_scale) as glcm_ASM")

        return glcm_features_df











