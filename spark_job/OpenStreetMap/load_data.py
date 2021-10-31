from pyspark.sql.functions import udf, explode
from pyspark.sql.types import IntegerType
from pyspark.sql.types import StringType
from pyspark.sql import functions as f


class LoadOSM:

    def __init__(self, path, spark):
        self.path = path
        self.spark = spark

    def getNodes(self):
        nodes = self.spark.read.parquet(self.path + "argentina-latest.osm.pbf.node.parquet")
        nodes = nodes.selectExpr("id", "tags", "CONCAT(longitude, ',',  latitude) as geomText")
        nodes = nodes.select("id", "geomText", "tags")
        # nodes = nodes.select("id", "geomText", explode("tags")).selectExpr("id",
        #                                                                    "ST_PointFromText(geomtext, ',') as Geometry",
        #                                                                    "col as attribute")
        return nodes

    @staticmethod
    def getlen(self, col):
        return len(col)

    @staticmethod
    def constructGeometry(self, col):
        temp = []
        col = sorted(col, key=lambda x: int(x[0]))
        first_geom = col[0][1]
        for arr in col:
            temp.append(arr[1])
        temp.append(first_geom)
        return ",".join(temp)

    @staticmethod
    def getString(hexa):
        return hexa.decode()

    def registerUDF(self):
        length = udf(self.getlen, IntegerType())
        self.spark.udf.register("ColLen", length)
        geom = udf(self.constructGeometry, StringType())
        self.spark.udf.register("GetGeom", geom)
        convert = udf(self.getString, StringType())
        self.spark.udf.register("RS_Convert", convert)


    def getWays(self):
        ways = self.spark.read.parquet(self.path + "argentina-latest.osm.pbf.way.parquet")
        ways = ways.select("id", "tags", explode("nodes")).select(
            "*", f.col("col")["index"].alias("index"), f.col("col")["nodeId"].alias("nodeId")
        )
        ways.createOrReplaceTempView("ways")
        nodes = self.getNodes()
        nodes.createOrReplaceTempView("points")
        waysJoinnodes = self.spark.sql(
            "Select ways.id as waysId,points.id as nodeId,ways.index as nodeindex, ways.tags as tags,points.geomText as geomText from points JOIN ways ON points.id=ways.nodeId")
        waysJoinnodes = waysJoinnodes.withColumn("index_geom", f.array(f.col("nodeindex"), f.col("geomText"))).select(
            "waysId", "nodeId", "tags", "index_geom")
        waysJoinnodes = waysJoinnodes.groupBy("waysId").agg(f.collect_set("index_geom").alias('array_geom'))
        self.registerUDF()
        waysJoinnodes = waysJoinnodes.selectExpr("waysId", "array_geom", "ColLen(array_geom) as length").where(
            "length>=3").selectExpr("waysId", "array_geom", "GetGeom(array_geom) as Geometry")
        waysJoinnodes.createOrReplaceTempView("waysjoin")
        finalways = self.spark.sql(
            "Select ways.id as id, ways.tags as tags, waysjoin.Geometry as Geometry from ways JOIN waysjoin ON ways.id=waysjoin.waysId ")
        finalways = finalways.distinct().select("id", explode("tags"), "Geometry")
        ways = finalways.selectExpr("id", "ST_PolygonFromText(Geometry, ',') as Geometry", "col as attribute")
        return ways

    def transform(self):
        nodes = self.getNodes()
        ways = self.getWays()


        nodes = nodes.select("id", "geomText", explode("tags")).selectExpr("id",
                                                                           "ST_PointFromText(geomtext, ',') as Geometry",
                                                                           "col as attribute")
        nodes = nodes.select(
            "*", f.col("attribute")["key"].alias("key"), f.col("attribute")["value"].alias("value"))
        # ways = ways.select(
        #     "*", f.col("attribute")["key"].alias("key"), f.col("attribute")["value"].alias("value"))
        # points = nodes.selectExpr("id", "Geometry", "RS_Convert(key) as attr_key",
        #                                        "RS_Convert(value) as attr_value")
        # polygons = ways.selectExpr("id", "Geometry", "RS_Convert(key) as attr_key", "RS_Convert(value) as attr_value")
        return nodes , ""



