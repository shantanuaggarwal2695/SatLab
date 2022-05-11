
class Loader:
    def __init__(self, path, spark):
        self.path = path
        self.spark = spark

    def load_geotiff(self):
        image_data = self.spark.read.format("geotiff").option("dropInvalid", True).load(self.path)
        image_data = image_data.selectExpr("image.origin as origin", "ST_GeomFromWkt(image.wkt) as Geom",
                                            "image.height as height", "image.width as width",
                                            "image.data as data", "image.nBands as bands")

        image_data = image_data.selectExpr("origin", "ST_Transform(Geom,'epsg:4326','epsg:3857') as Geom", "height",
                                             "width", "data", "bands")

        return image_data



