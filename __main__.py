from pyspark.sql import SparkSession
from sedona.register import SedonaRegistrator
from sedona.utils import KryoSerializer, SedonaKryoRegistrator



def initiate_session():
    spark = SparkSession. \
        builder. \
        appName("geo_labeling_functions"). \
        config("spark.serializer", KryoSerializer.getName). \
        config("spark.kryo.registrator", SedonaKryoRegistrator.getName). \
        config("spark.driver.memory", "20g"). \
        config("spark.executor.memory", "64g"). \
        config("spark.sql.crossJoin.enabled", "true"). \
        getOrCreate()

    # config("spark.driver.maxResultSize", "5g"). \

    SedonaRegistrator.registerAll(spark)
    sc = spark.sparkContext
    sc.addPyFile("/hdd2/shantanuCodeData/SatLab/dist/experiment-0.1.0-py3.7.egg")

    return spark

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


if __name__ == '__main__':

    spark = initiate_session()
    from spark_job.Loading.geotiff_loader import Loader
    from spark_job.OpenStreetMap.load_data import LoadOSM
    from spark_job.Features.spatial import SpatialFunctions
    from spark_job.Features.textural import Textural
    from spark_job.Labeling.manual import Manual
    from spark_job.Labeling.semi import SemiLabeling
    from spark_job.Labeling.automatic import Automatic

    loader = Loader("/hdd2/shantanuCodeData/data/scalability_test/set_1/", spark)
    train = loader.load_geotiff()
    new_train = train.coalesce(5000)
    new_train.persist().show()
    print(new_train.count())


    # OSM = LoadOSM("/hdd2/shantanuCodeData/data/pbf/slum_data/", spark)
    # points, polygons = OSM.transform()
    # points.persist().show()
    # polygons.persist().show()

    # spatialfunctions = SpatialFunctions(points, polygons, new_train, spark)
    # geo_features = spatialfunctions.combine()
    # geo_features.persist()
    # points.unpersist()
    # polygons.unpersist()

    texturalfunctions = Textural(new_train, spark)
    glcm_df = texturalfunctions.extract_features()
    glcm_df = glcm_df.selectExpr("origin", "ST_AsText(Geom) as Geom", "glcm_contrast_Scaled",
                            "glcm_dissimilarity_Scaled", "glcm_homogeneity_Scaled",
                            "glcm_energy_Scaled", "glcm_correlation_Scaled", "glcm_ASM_Scaled")
    new_train.unpersist()
    # glcm_df.persist()

    glcm_df.show()
    glcm_df.printSchema()

    # geo_features.show()
    # glcm_df.show()
    #
    # geo_features.write.format("csv").save("/hdd2/shantanuCodeData/data/experiments/features/spatial")
    glcm_df.write.format("csv").save("/hdd2/shantanuCodeData/data/experiments/features/textural/image")


    # ManualLabeling = Manual(geo_features, glcm_df, spark)
    # labels = ManualLabeling.produce_labels()
    # print(type(labels))
    # print(labels)

    # Semi = SemiLabeling(geo_features, glcm_df, spark)
    # labels = Semi.generate_class()
    # print(labels)

    # AutoLabel = Automatic(geo_features, glcm_df, spark)
    # rules = AutoLabel.generate_rules(4)
    # print(rules)


