# from Loading.geotiff_loader import *
# from OpenStreetMap.load_data import *


# def initiate_session():
#     spark = SparkSession. \
#         builder. \
#         master("spark://EN4119507L.cidse.dhcp.asu.edu:7077"). \
#         appName("geo_labeling_functions"). \
#         config("spark.serializer", KryoSerializer.getName). \
#         config("spark.kryo.registrator", SedonaKryoRegistrator.getName). \
#         config("spark.driver.memory", "10g"). \
#         config("spark.executor.memory", "15g"). \
#         config("spark.driver.maxResultSize", "5g"). \
#         config("spark.network.timeout", "1000s"). \
#         config("spark.kryoserializer.buffer.max", "1024"). \
#         config("spark.sql.broadcastTimeout", "36000"). \
#         config("spark.sql.crossJoin.enabled", "true"). \
#         getOrCreate()
#     SedonaRegistrator.registerAll(spark)
#     sc = spark.sparkContext
#     sc.addPyFile("/hdd2/shantanuCodeData/SatLab/dist/experiment-0.1.0-py3.7.egg")
#
#     return spark



# def run_experiment(spark):
#     loader = Loader("/hdd2/shantanuCodeData/data/manual_audit/", spark)
#     train = loader.load_geotiff()
#     train.show(2)
#
#     # Prepare OSM data
#     OSM = LoadOSM("/hdd2/shantanuCodeData/data/pbf/slum_data/", spark)
#     points, polygons = OSM.transform()
#     points.show(2)
#     polygons.show(2)





