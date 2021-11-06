import numpy as np
from pyspark.sql.types import DoubleType
from skimage.feature import greycomatrix
from skimage.feature import greycoprops as gc


def GLCM_Contrast(pixels):
    mi, ma = 0, 255
    ks = 5
    h,w = 32,32
    nbit = 8
    bins = np.linspace(mi, ma+1, nbit+1)
    bin_image = np.digitize(pixels, bins) - 1
    new_image = np.reshape(bin_image, (32,32))
    glcm = greycomatrix(new_image, [1], [0, np.pi/4, np.pi/2], levels=8 , normed=True, symmetric=True)
    feature = gc(glcm, "contrast")
    return sum(feature[0].tolist())/len(feature[0].tolist())

def GLCM_Dissimilarity(pixels):
    mi, ma = 0, 255
    ks = 5
    h,w = 32,32
    nbit = 8
    bins = np.linspace(mi, ma+1, nbit+1)
    bin_image = np.digitize(pixels, bins) - 1
    new_image = np.reshape(bin_image, (32,32))
    glcm = greycomatrix(new_image, [1], [0, np.pi/4, np.pi/2], levels=8 , normed=True, symmetric=True)
    feature = gc(glcm, "dissimilarity")
    return sum(feature[0].tolist())/len(feature[0].tolist())


def GLCM_Homogeneity(pixels):
    mi, ma = 0, 255
    ks = 5
    h,w = 32,32
    nbit = 8
    bins = np.linspace(mi, ma+1, nbit+1)
    bin_image = np.digitize(pixels, bins) - 1
    new_image = np.reshape(bin_image, (32,32))
    glcm = greycomatrix(new_image, [1], [0, np.pi/4, np.pi/2], levels=8 , normed=True, symmetric=True)
    feature = gc(glcm, "homogeneity")
    return sum(feature[0].tolist())/len(feature[0].tolist())

def GLCM_Energy(pixels):
    mi, ma = 0, 255
    ks = 5
    h,w = 32,32
    nbit = 8
    bins = np.linspace(mi, ma+1, nbit+1)
    bin_image = np.digitize(pixels, bins) - 1
    new_image = np.reshape(bin_image, (32,32))
    glcm = greycomatrix(new_image, [1], [0, np.pi/4, np.pi/2], levels=8 , normed=True, symmetric=True)
    feature = gc(glcm, "energy")
    return sum(feature[0].tolist())/len(feature[0].tolist())


def GLCM_Correlation(pixels):
    mi, ma = 0, 255
    ks = 5
    h,w = 32,32
    nbit = 8
    bins = np.linspace(mi, ma+1, nbit+1)
    bin_image = np.digitize(pixels, bins) - 1
    new_image = np.reshape(bin_image, (32,32))
    glcm = greycomatrix(new_image, [1], [0, np.pi/4, np.pi/2], levels=8 , normed=True, symmetric=True)
    feature = gc(glcm, "correlation")
    return sum(feature[0].tolist())/len(feature[0].tolist())

def GLCM_ASM(pixels):
    mi, ma = 0, 255
    ks = 5
    h,w = 32,32
    nbit = 8
    bins = np.linspace(mi, ma+1, nbit+1)
    bin_image = np.digitize(pixels, bins) - 1
    new_image = np.reshape(bin_image, (32,32))
    glcm = greycomatrix(new_image, [1], [np.pi/4,np.pi/2], levels=8 , normed=True, symmetric=True)
    feature = gc(glcm, "ASM")
    return sum(feature[0].tolist())/len(feature[0].tolist())


def register_udf(spark):
    spark.udf.register("GLCM_Contrast", GLCM_Contrast,DoubleType())
    spark.udf.register("GLCM_Dis", GLCM_Dissimilarity, DoubleType())
    spark.udf.register("GLCM_Hom", GLCM_Homogeneity, DoubleType())
    spark.udf.register("GLCM_ASM", GLCM_ASM, DoubleType())
    spark.udf.register("GLCM_Corr", GLCM_Correlation, DoubleType())
    spark.udf.register("GLCM_Energy", GLCM_Energy, DoubleType())
