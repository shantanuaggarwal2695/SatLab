Run the egg file using:
spark-submit --master spark://EN4119508L.cidse.dhcp.asu.edu:7077 --py-files dist/experiment-0.1.0-py3.7.egg __main__.py


Create an egg file:
python setup.py bdist_egg



1. spark_job/Features/texture.py: It contains textural and spatial feature extraqction scripts for satellite images that makes use of map algebra operators, UDF in apache sedona for implementing GLCM matrix and it's metrics

2. spark_job/Features/spatial.py: It contains multiple ways to extract spatial features using images and OpenStreetmap data with a help of spatial joins.

3. OpenStreetMap: It contains scripts for loading OSM data in Apache sedona and query them using simple Spark SQL operators.

4. Loading: It shows ways to load GeoTIFF images in a GeoTIFF dataframe with a help of scala/java API in Apache Sedona which is also our conttribution

5. Labeling: It contains three different modes of labeling (1)manual: user needs to fix heuristics and thresholds (2) semi-automatic: user just needs to give heuristics and we find out thresholds using cluster similarity (3) automatic mode: We will output ensemble of labeling functions using random forest trained with depth of 3 and 5 trees on 4 different partitions.

We have also contributed GeoTIFF loader and map algebra operators in Apache Sedona: [SEDONA-30] Add raster data support in Sedona SQL (#523): https://github.com/apache/incubator-sedona/commit/8fd688f4c26374bf2f4811d1dc4c333d9acd3b4d
