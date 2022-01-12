from pyspark.sql import SparkSession
from sedona.register import SedonaRegistrator
from sedona.utils import KryoSerializer, SedonaKryoRegistrator



def initiate_session():
    spark = SparkSession. \
        builder. \
        appName("vizTest"). \
        master("spark://EN4119508L.cidse.dhcp.asu.edu:7077").\
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
    # sc.addPyFile("/hdd2/shantanuCodeData/SatLab/dist/experiment-0.1.0-py3.7.egg")

    return spark


def run_job(path,spark,index_list=None):
    from spark_job.Loading.geotiff_loader import Loader
    from spark_job.OpenStreetMap.load_data import LoadOSM
    from spark_job.Features.spatial import SpatialFunctions
    from spark_job.Features.textural import Textural
    from spark_job.Labeling.manual import Manual
    from spark_job.Labeling.semi import SemiLabeling
    from spark_job.Labeling.automatic import Automatic

    loader = Loader(path, spark)
    train = loader.load_geotiff()

    OSM = LoadOSM("/hdd2/shantanuCodeData/data/pbf/slum_data/", spark)
    points, polygons = OSM.transform()

    spatialfunctions = SpatialFunctions(points, polygons, train, spark)
    geo_features = spatialfunctions.combine()

    texturalfunctions = Textural(train, spark)
    glcm_df = texturalfunctions.extract_features()

    ManualLabeling = Manual(geo_features, glcm_df, spark, index_list)
    labels = ManualLabeling.produce_labels()
    # print(type(labels))
    print(labels)

    # Semi = SemiLabeling(geo_features, glcm_df, spark)
    # labels = Semi.generate_class()
    # print(labels)

    # AutoLabel = Automatic(geo_features, glcm_df, spark)
    # rules = AutoLabel.generate_rules(4)
    # print(rules)


# if __name__ == '__main__':
#
#     spark = initiate_session()
#     run_job(spark)

