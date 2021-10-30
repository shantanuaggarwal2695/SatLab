import findspark
findspark.find()

from pyspark.sql import SparkSession
from sedona.utils import SedonaKryoRegistrator, KryoSerializer
from sedona.register import SedonaRegistrator
import os
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
    sc = spark.sparkContext
    sc.addPyFile("/hdd2/shantanuCodeData/SatLab/dist/experiment-0.1.0-py3.7.egg")

    return spark


if __name__ == '__main__':

    # os.environ['SPARK_HOME'] = "/hdd2/shantanuCodeData/lib/spark-3.1.1-bin-hadoop2.7"
    spark = initiate_session()
    from spark_job import start
    start.run_experiment(spark)