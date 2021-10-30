from pyspark.sql import SparkSession
from sedona.utils import SedonaKryoRegistrator, KryoSerializer
from sedona.register import SedonaRegistrator
import os



if __name__ == '__main__':


    from spark_job import start

    spark = start.initiate_session()
    start.run_experiment(spark)