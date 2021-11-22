from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

class SpatialFunctions:
    def __init__(self, points, polygons, train, spark):
        self.osm_points = points
        self.osm_polygons = polygons
        self.spark = spark
        self.train = train
        self.train.createOrReplaceTempView("train")
        bs_df = self.spark.sql(
            "Select ST_Transform(ST_PointFromText('-58.381592,-34.603722',','), 'epsg:4326','epsg:3857') as Geom")
        bs_df.createOrReplaceTempView("bs")

    def healthcare(self):
        healthcare_points = self.osm_points.filter((self.osm_points.attr_key == "healthcare") & (self.osm_points.attr_value == "hospital"))

        healthcare_polygons = self.osm_polygons.filter((self.osm_polygons.attr_key == "healthcare") & (self.osm_polygons.attr_value == "hospital")).selectExpr("id",
                                                                                              "ST_Centroid(Geometry) as Geometry",
                                                                                              "attr_key", "attr_value")
        healthcare_data = healthcare_points.union(healthcare_polygons)
        healthcare_data = healthcare_data.selectExpr("id", "ST_Transform(Geometry, 'epsg:4326','epsg:3857') as Geometry",
                                                 "attr_key", "attr_value")
        healthcare_data.createOrReplaceTempView("temp_healthcare")

        max_distance = self.spark.sql(
            "Select train.Geom, bs.Geom as fixed_Geom,ST_Distance(train.Geom, bs.Geom)/1000 as distance from train, bs ")
        max_distance.createOrReplaceTempView("distance_view")
        max_distance_ = self.spark.sql(
            "Select Geom, fixed_Geom , distance from distance_view where distance = (Select max(distance) from distance_view)")
        max_distance_.show()
        max_distance_.createOrReplaceTempView("max_geom_distance")
        nearest_healthcare_points = self.spark.sql(
            "SELECT temp_healthcare.Geometry, ST_Distance(temp_healthcare.Geometry, bs.Geom)/1000 AS distance FROM temp_healthcare,bs where ST_Distance(temp_healthcare.Geometry, bs.Geom)/1000<=185")
        nearest_healthcare_points.createOrReplaceTempView("new_healthcare")

        # Spatial Feature Extraction
        hpoints2 = self.spark.sql(
            "Select train.origin, train.Geom, new_healthcare.Geometry, ST_Distance(train.Geom,new_healthcare.Geometry)/1000 as distance from train,new_healthcare")
        hpoints2.createOrReplaceTempView("hp2")
        hpoints2 = self.spark.sql(
            "Select origin, first(hp2.Geom) as Geom, AVG(hp2.distance) as distance_health from hp2 GROUP BY origin")
        return hpoints2


    def shopping_malls(self):
        polygons_malls = self.osm_polygons.filter((self.osm_polygons.attr_key == "shop") & (self.osm_polygons.attr_value == "mall"))
        points_malls = self.osm_points.filter((self.osm_points.attr_key == "shop") & (self.osm_points.attr_value == "mall"))

        mall_data = points_malls.union(polygons_malls)
        mall_data = mall_data.selectExpr("id", "ST_Transform(Geometry, 'epsg:4326','epsg:3857') as Geometry",
                                         "attr_key", "attr_value")
        mall_data = mall_data.repartition(2000)
        mall_data.persist()
        mall_data.createOrReplaceTempView("temp_malls")

        nearest_mall_points = self.spark.sql(
            "SELECT temp_malls.Geometry, ST_Distance(temp_malls.Geometry, bs.Geom)/1000 AS distance FROM temp_malls,bs where ST_Distance(temp_malls.Geometry, bs.Geom)/1000 < 185 ")
        nearest_mall_points.createOrReplaceTempView("new_malls")
        nearest_mall_points.persist()

        # Features
        labeling_function_6 = self.spark.sql(
            "Select train.origin, train.Geom, new_malls.Geometry, ST_Distance(train.Geom,new_malls.Geometry)/1000 as distance from train,new_malls")
        labeling_function_6.createOrReplaceTempView("lf6")
        labeling_function_6.persist()
        labeling_function_6 = self.spark.sql(
            "Select origin, first(lf6.Geom) as Geom, AVG(lf6.distance) as distance_malls from lf6 GROUP BY origin")
        return labeling_function_6

    def schools(self):
        polygons_schools = self.osm_polygons.filter(((self.osm_polygons.attr_key == "building") & (self.osm_polygons.attr_value == "school")) | (
                    (self.osm_polygons.attr_key == "amenity") & (self.osm_polygons.attr_value == "school")))
        points_school = self.osm_points.filter(((self.osm_points.attr_key == "building") & (self.osm_points.attr_value == "school")) | (
                    (self.osm_points.attr_key == "amenity") & (self.osm_points.attr_value == "school")))
        school_data = points_school.union(polygons_schools)
        school_data = school_data.selectExpr("id", "ST_Transform(Geometry, 'epsg:4326','epsg:3857') as Geometry",
                                             "attr_key", "attr_value")
        school_data.persist()
        school_data.createOrReplaceTempView("temp_schools")
        nearest_school_points = self.spark.sql(
            "SELECT temp_schools.Geometry, ST_Distance(temp_schools.Geometry, bs.Geom)/1000 AS distance FROM temp_schools,bs where ST_Distance(temp_schools.Geometry, bs.Geom)/1000 <= 185")
        nearest_school_points.createOrReplaceTempView("schools_nearest")
        nearest_school_points.persist()

        labeling_function_7 = self.spark.sql(
            "Select train.origin, train.Geom, schools_nearest.Geometry, ST_Distance(train.Geom,schools_nearest.Geometry)/1000 as distance from train,schools_nearest")
        labeling_function_7.createOrReplaceTempView("lf7")
        labeling_function_7.persist()
        labeling_function_7 = self.spark.sql(
            "Select origin, first(lf7.Geom) as Geom, AVG(lf7.distance) as distance_schools from lf7 GROUP BY origin")
        return labeling_function_7

    def waste(self):
        polygon_waste_dispsal = self.osm_polygons.filter((self.osm_polygons.attr_key == "landuse") & (self.osm_polygons.attr_value == "landfill"))
        points_waste_disposal = self.osm_points.filter((self.osm_points.attr_key == "landuse") & (self.osm_points.attr_value == "landfill"))
        waste_data = polygon_waste_dispsal.union(points_waste_disposal)
        waste_data = waste_data.selectExpr("id", "ST_Transform(Geometry, 'epsg:4326','epsg:3857') as Geometry",
                                           "attr_key", "attr_value")
        waste_data.persist()
        waste_data.createOrReplaceTempView("temp_waste")

        nearest_waste_points = self.spark.sql(
            "SELECT temp_waste.Geometry, ST_Distance(temp_waste.Geometry, bs.Geom)/1000 AS distance FROM temp_waste,bs where ST_Distance(temp_waste.Geometry, bs.Geom)/1000<=185 ")
        nearest_waste_points.createOrReplaceTempView("waste_nearest")
        nearest_waste_points.show()
        nearest_waste_points.persist()
        labeling_function_8 = self.spark.sql(
            "Select train.origin, train.Geom, waste_nearest.Geometry, ST_Distance(train.Geom,waste_nearest.Geometry)/1000 as distance from train,waste_nearest")
        labeling_function_8.createOrReplaceTempView("lf8")
        labeling_function_8.persist()
        labeling_function_8 = self.spark.sql(
            "Select origin,first(lf8.Geom) as Geom,MIN(lf8.distance) as distance_waste from lf8 GROUP BY origin")
        return labeling_function_8

    def roads(self):
        polygon_paved = self.osm_polygons.filter((self.osm_polygons.attr_key == "surface") & (self.osm_polygons.attr_value == "paved"))
        points_paved = self.osm_points.filter((self.osm_points.attr_key == "surface") & (self.osm_points.attr_value == "paved"))
        paved_data = polygon_paved.union(points_paved)
        paved_data = paved_data.selectExpr("id", "ST_Transform(Geometry, 'epsg:4326','epsg:3857') as Geometry",
                                           "attr_key", "attr_value")
        paved_data.persist()
        paved_data.createOrReplaceTempView("paved_temp")

        nearest_road_points = self.spark.sql(
            "SELECT paved_temp.Geometry, ST_Distance(paved_temp.Geometry, bs.Geom)/1000 AS distance FROM paved_temp,bs where ST_Distance(paved_temp.Geometry, bs.Geom)/1000<=185")
        nearest_road_points.createOrReplaceTempView("paved_nearest")
        nearest_road_points.persist()
        labeling_function_9 = self.spark.sql(
            "Select train.origin, train.Geom, paved_nearest.Geometry, ST_Distance(train.Geom,paved_nearest.Geometry)/1000 as distance from train,paved_nearest")
        labeling_function_9.createOrReplaceTempView("lf9")
        labeling_function_9.persist()
        labeling_function_9 = self.spark.sql(
            "Select origin, first(lf9.Geom) as Geom, AVG(lf9.distance) as distance_road from lf9 GROUP BY origin")
        return labeling_function_9

    def forest(self):
        polygon_forest = self.osm_polygons.filter((self.osm_polygons.attr_key == "landuse") & (self.osm_polygons.attr_value == "forest"))
        points_forest = self.osm_points.filter((self.osm_points.attr_key == "landuse") & (self.osm_points.attr_value == "forest"))
        forest_data = polygon_forest.union(points_forest)
        forest_data = forest_data.selectExpr("id", "ST_Transform(Geometry, 'epsg:4326','epsg:3857') as Geometry",
                                             "attr_key", "attr_value")

        forest_data.persist()
        forest_data.createOrReplaceTempView("forest_temp")

        nearest_forest_points = self.spark.sql(
            "SELECT forest_temp.Geometry, ST_Distance(forest_temp.Geometry, bs.Geom)/1000 AS distance FROM forest_temp,bs where ST_Distance(forest_temp.Geometry, bs.Geom)/1000<=185")
        nearest_forest_points.createOrReplaceTempView("forest_nearest")
        nearest_forest_points.persist()

        labeling_function_10 = self.spark.sql(
            "Select train.origin, train.Geom, forest_nearest.Geometry, ST_Distance(train.Geom,forest_nearest.Geometry) as distance from train,forest_nearest")
        labeling_function_10.persist()

        labeling_function_10.createOrReplaceTempView("lf10")
        labeling_function_10 = self.spark.sql(
            "Select origin, first(lf10.Geom) as Geom, MIN(lf10.distance) as distance_forest from lf10 GROUP BY origin")
        return labeling_function_10

    def residential(self):
        polygon_res = self.osm_polygons.filter((self.osm_polygons.attr_key == "landuse") & (self.osm_polygons.attr_value == "residential"))
        points_res = self.osm_points.filter((self.osm_points.attr_key == "landuse") & (self.osm_points.attr_value == "residential"))
        res_data = polygon_res.union(points_res)
        res_data = res_data.selectExpr("id", "ST_Transform(Geometry, 'epsg:4326','epsg:3857') as Geometry", "attr_key",
                                       "attr_value")

        res_data.persist()
        res_data.createOrReplaceTempView("res_temp")

        nearest_res_points = self.spark.sql(
            "SELECT res_temp.Geometry, ST_Distance(res_temp.Geometry, bs.Geom)/1000 AS distance FROM res_temp,bs where ST_Distance(res_temp.Geometry, bs.Geom)/1000<=185")
        nearest_res_points.createOrReplaceTempView("res_nearest")
        nearest_res_points.persist()

        labeling_function_11 = self.spark.sql(
            "Select train.origin, train.Geom, res_nearest.Geometry, ST_Distance(train.Geom,res_nearest.Geometry) as distance from train,res_nearest")
        labeling_function_11.createOrReplaceTempView("lf11")
        labeling_function_11.persist()
        labeling_function_11 = self.spark.sql(
            "Select origin, first(lf11.Geom) as Geom,AVG(lf11.distance) as distance_res from lf11 GROUP BY origin")
        return labeling_function_11

    def power(self):
        polygon_power = self.osm_polygons.filter((self.osm_polygons.attr_key == "power") & (self.osm_polygons.attr_value == "line"))
        points_power = self.osm_points.filter((self.osm_points.attr_key == "power") & (self.osm_points.attr_value == "line"))
        power_data = polygon_power.union(points_power)
        power_data = power_data.selectExpr("id", "ST_Transform(Geometry, 'epsg:4326','epsg:3857') as Geometry",
                                           "attr_key", "attr_value")
        power_data.createOrReplaceTempView("power_temp")

        nearest_pow_points = self.spark.sql(
            "SELECT power_temp.Geometry, ST_Distance(power_temp.Geometry, bs.Geom)/1000 AS distance FROM power_temp,bs where ST_Distance(power_temp.Geometry, bs.Geom)/1000<=185 ")
        nearest_pow_points.createOrReplaceTempView("pow_nearest")
        nearest_pow_points.persist()

        labeling_function_12 = self.spark.sql(
            "Select train.origin, train.Geom, pow_nearest.Geometry, ST_Distance(train.Geom,pow_nearest.Geometry) as distance from train,pow_nearest")
        labeling_function_12.createOrReplaceTempView("lf12")
        labeling_function_12.persist()
        labeling_function_12 = self.spark.sql(
            "Select origin, first(lf12.Geom) as Geom, AVG(lf12.distance) as distance_pow from lf12 GROUP BY origin")
        return labeling_function_12

    def resorts(self):
        polygon_resort = self.osm_polygons.filter((self.osm_polygons.attr_key == "leisure") & (self.osm_polygons.attr_value == "resort"))
        points_resort = self.osm_points.filter((self.osm_points.attr_key == "leisure") & (self.osm_points.attr_value == "resort"))
        resort_data = polygon_resort.union(points_resort)

        polygon_pool = self.osm_polygons.filter((self.osm_polygons.attr_key == "leisure") & (self.osm_polygons.attr_value == "swimming_pool"))
        points_pool = self.osm_points.filter((self.osm_points.attr_key == "leisure") & (self.osm_points.attr_value == "swimming_pool"))
        pool_data = polygon_pool.union(points_pool)

        recreat_data = resort_data.union(pool_data)
        recreat_data = recreat_data.selectExpr("id", "ST_Transform(Geometry, 'epsg:4326','epsg:3857') as Geometry",
                                               "attr_key", "attr_value")
        recreat_data.persist()
        recreat_data.createOrReplaceTempView("recreat_temp")
        nearest_recreat_points = self.spark.sql(
            "SELECT recreat_temp.Geometry, ST_Distance(recreat_temp.Geometry, bs.Geom)/1000 AS distance FROM recreat_temp,bs where ST_Distance(recreat_temp.Geometry, bs.Geom)/1000<=185 ")
        nearest_recreat_points.createOrReplaceTempView("recreat_nearest")
        nearest_recreat_points.persist()
        labeling_function_13 = self.spark.sql(
            "Select train.origin, train.Geom, recreat_nearest.Geometry, ST_Distance(train.Geom,recreat_nearest.Geometry) as distance from train,recreat_nearest")
        labeling_function_13.createOrReplaceTempView("lf13")
        labeling_function_13.persist()
        labeling_function_13 = self.spark.sql(
            "Select origin, first(lf13.Geom) as Geom, AVG(lf13.distance) as distance_resort from lf13 GROUP BY origin")
        return labeling_function_13

    def grasslands(self):
        polygon_grasslands = self.osm_polygons.filter((self.osm_polygons.attr_key == "natural") & (self.osm_polygons.attr_value == "grassland"))
        points_grasslands = self.osm_points.filter((self.osm_points.attr_key == "natural") & (self.osm_points.attr_value == "grassland"))
        grassland_data = polygon_grasslands.union(points_grasslands)

        polygon_meadows = self.osm_polygons.filter((self.osm_polygons.attr_key == "landuse") & (self.osm_polygons.attr_value == "meadow"))
        points_meadows = self.osm_points.filter((self.osm_points.attr_key == "landuse") & (self.osm_points.attr_value == "meadow"))
        meadows_data = polygon_meadows.union(points_meadows)
        green_data = grassland_data.union(meadows_data)
        green_data = green_data.selectExpr("id", "ST_Transform(Geometry, 'epsg:4326','epsg:3857') as Geometry",
                                           "attr_key", "attr_value")
        green_data.show()
        green_data.persist()
        green_data.createOrReplaceTempView("green_temp")

        nearest_green_points = self.spark.sql(
            "SELECT green_temp.Geometry, ST_Distance(green_temp.Geometry, bs.Geom)/1000 AS distance FROM green_temp,bs where ST_Distance(green_temp.Geometry, bs.Geom)/1000<=185 ")
        nearest_green_points.createOrReplaceTempView("green_nearest")
        nearest_green_points.persist()

        labeling_function_14 = self.spark.sql(
            "Select train.origin, train.Geom, green_nearest.Geometry, ST_Distance(train.Geom,green_nearest.Geometry) as distance from train,green_nearest")
        labeling_function_14.createOrReplaceTempView("lf14")
        labeling_function_14.persist()
        labeling_function_14 = self.spark.sql(
            "Select origin,first(lf14.Geom) as Geom, AVG(lf14.distance) as distance_green from lf14 GROUP BY origin")
        return labeling_function_14

    def combine(self):
        self.healthcare().createOrReplaceTempView("domain1")
        self.shopping_malls().createOrReplaceTempView("domain2")
        self.schools().createOrReplaceTempView("domain3")
        self.waste().createOrReplaceTempView("domain4")
        self.roads().createOrReplaceTempView("domain5")
        self.forest().createOrReplaceTempView("domain6")
        self.residential().createOrReplaceTempView("domain7")
        self.power().createOrReplaceTempView("domain8")
        self.resorts().createOrReplaceTempView("domain9")
        self.grasslands().createOrReplaceTempView("domain10")
        geo_features = self.spark.sql(

            "select domain1.origin, domain1.Geom, domain1.distance_health as healthcare, domain2.distance_malls as malls, domain3.distance_schools as schools, domain4.distance_waste as waste, domain5.distance_road as road, domain6.distance_forest as forest, domain7.distance_res as residential, domain8.distance_pow as power, domain9.distance_resort as resort, domain10.distance_green as greenland from domain1 JOIN domain2 JOIN domain3 JOIN domain4 JOIN domain5 JOIN domain6 JOIN domain7 JOIN domain8 JOIN domain9 JOIN domain10 ON (domain1.Geom = domain2.Geom AND domain1.Geom = domain3.Geom AND domain1.Geom = domain4.Geom AND domain1.Geom = domain5.Geom AND domain1.Geom = domain6.Geom AND domain1.Geom = domain7.Geom AND domain1.Geom = domain8.Geom AND domain1.Geom=domain9.Geom AND domain1.Geom=domain10.Geom)")
        geo_features.persist()

        unlist = udf(lambda x: round(float(list(x)[0]), 3), DoubleType())
        for i in ["healthcare", "malls", "schools", "waste", "road", "forest", "residential", "power", "resort",
                  "greenland"]:
            # VectorAssembler Transformation - Converting column to vector type
            assembler = VectorAssembler(inputCols=[i], outputCol=i + "_Vect")

            # MinMaxScaler Transformation
            scaler = MinMaxScaler(inputCol=i + "_Vect", outputCol=i + "_Scaled")

            # Pipeline of VectorAssembler and MinMaxScaler
            pipeline = Pipeline(stages=[assembler, scaler])

            # Fitting pipeline on dataframe
            geo_features = pipeline.fit(geo_features).transform(geo_features).withColumn(i + "_Scaled",
                                                                                         unlist(i + "_Scaled")).drop(
                i + "_Vect")

        print("After Scaling :")
        geo_features = geo_features.selectExpr("origin", "Geom", "healthcare_Scaled as healthcare",
                                               "malls_Scaled as malls", "schools_Scaled as schools",
                                               "waste_Scaled as waste", "road_Scaled as road",
                                               "forest_Scaled as forest", "residential_Scaled as residential",
                                               "power_Scaled as power", "resort_Scaled as resort",
                                               "greenland_Scaled as grasslands")
        return geo_features

































