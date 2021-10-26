import os
from IPython.display import display, HTML
from pyspark.sql import SparkSession
from pyspark import StorageLevel
import geopandas as gpd
import pandas as pd
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import LongType
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql.functions import regexp_replace
from sedona.register import SedonaRegistrator
from sedona.utils import SedonaKryoRegistrator, KryoSerializer
from sedona.utils import SedonaKryoRegistrator, KryoSerializer
#from pyspark.sql.functions import udf, lit, explode
from pyspark import StorageLevel
import seaborn as sns
from sedona.utils.adapter import Adapter
from sedona.core.spatialOperator import KNNQuery
from shapely.geometry import Point
import pyspark.sql.functions as f
from Loading.geotiff_loader import *


def initiate_session():
    spark = SparkSession. \
        builder. \
        master("spark://EN4119507L.cidse.dhcp.asu.edu:7077"). \
        appName("geo_labeling_functions"). \
        config("spark.serializer", KryoSerializer.getName). \
        config("spark.kryo.registrator", SedonaKryoRegistrator.getName). \
        config("spark.driver.memory", "10g"). \
        config("spark.executor.memory", "15g"). \
        config("spark.driver.maxResultSize", "5g"). \
        config("spark.network.timeout", "1000s"). \
        config("spark.kryoserializer.buffer.max", "1024"). \
        config("spark.sql.broadcastTimeout", "36000"). \
        config("spark.sql.crossJoin.enabled", "true"). \
        getOrCreate()
    SedonaRegistrator.registerAll(spark)
    # sc = spark.sparkContext

    return spark


if __name__ == '__main__':
    spark = initiate_session()

    loader = Loader("/hdd2/shantanuCodeData/data/manual_audit/", spark)
    train = loader.load_geotiff()
    train.show(2)

