from snorkel.labeling.lf import labeling_function
import logging
import pyspark.sql.functions as F
from pyspark import SparkContext
from pyspark.sql import SQLContext
from snorkel.labeling.model import LabelModel
from snorkel.labeling.apply.spark import SparkLFApplier

thresholds = [0]*16
class Manual:
    def __init__(self, geo_df, text_df, spark, index_list):
        self.geo = geo_df
        self.text = text_df
        self.geo.createOrReplaceTempView("geo");
        self.text.createOrReplaceTempView("text")
        self.spark = spark
        self.index_list = index_list
        self.combined_df = self.spark.sql(
            "select geo.origin, geo.Geom, geo.healthcare, geo.malls, geo.schools, geo.waste, geo.road, geo.forest, geo.residential, geo.power, geo.resort, geo.grasslands,text.glcm_contrast_Scaled, text.glcm_dissimilarity_Scaled, text.glcm_homogeneity_Scaled, text.glcm_energy_Scaled, text.glcm_correlation_Scaled, text.glcm_ASM_Scaled from text, geo where text.origin = geo.origin")

    def construct_labeling(self):

        SLUM = 1
        NON_SLUM = 0

        @labeling_function()
        def lf1(x):
            # 0.1
            if x.healthcare < thresholds[0]:
                return NON_SLUM
            else:
                return SLUM

        @labeling_function()
        def lf2(x):
            # 0.11
            if x.malls < thresholds[1]:
                return NON_SLUM
            else:
                return SLUM

        @labeling_function()
        def lf3(x):
            # 0.12
            if x.schools < thresholds[2]:
                return NON_SLUM
            else:
                return SLUM

        @labeling_function()
        def lf4(x):
            # 0.1
            if x.waste < thresholds[3]:
                return SLUM
            else:
                return NON_SLUM

        @labeling_function()
        def lf5(x):
            #0.07
            if x.forest < thresholds[4]:
                return SLUM
            else:
                return NON_SLUM

        @labeling_function()
        def lf6(x):
            # 0.05
            if x.power < thresholds[5]:
                return NON_SLUM
            else:
                return SLUM

        @labeling_function()
        def lf7(x):
            # 0.2
            if x.residential < thresholds[6]:
                return NON_SLUM
            else:
                return SLUM

        @labeling_function()
        def lf8(x):
            # 0.05
            if x.road < thresholds[7]:
                return NON_SLUM
            else:
                return SLUM

        @labeling_function()
        def lf9(x):
            # 0.07
            if x.resort < thresholds[8]:
                return NON_SLUM
            else:
                return SLUM

        @labeling_function()
        def lf10(x):

            #0.57
            if x.grasslands < thresholds[9]:
                return SLUM
            else:
                return NON_SLUM

        @labeling_function()
        def lf11(x):
            #0.8
            if x.glcm_contrast_Scaled < thresholds[10]:
                return SLUM
            else:
                return NON_SLUM

        @labeling_function()
        def lf12(x):
            #0.65
            if x.glcm_dissimilarity_Scaled < thresholds[11]:
                return SLUM
            else:
                return NON_SLUM

        @labeling_function()
        def lf13(x):
            # 0.25
            if x.glcm_homogeneity_Scaled < thresholds[12]:
                return NON_SLUM
            else:
                return SLUM

        @labeling_function()
        def lf14(x):

            # 0.10
            if x.glcm_energy_Scaled <= thresholds[13]:
                return NON_SLUM
            else:
                return SLUM

        @labeling_function()
        def lf15(x):

            #0.2
            if x.glcm_correlation_Scaled <= thresholds[14]:
                return NON_SLUM
            else:
                return SLUM

        @labeling_function()
        def lf16(x):

            #0.6
            if x.glcm_ASM_Scaled < thresholds[15]:
                return NON_SLUM
            else:
                return SLUM

        return [lf1, lf2, lf3, lf4, lf5, lf6, lf7, lf8, lf9, lf10, lf11, lf12, lf13, lf14, lf15, lf16]

    def produce_labels(self):

        for tuple in self.index_list:
            thresholds[tuple[0]-1] = tuple[1]

        lfs = self.construct_labeling()
        # Based on indexes from the user
        # lfs = [lfs[i-1] for i in self.index_list]
        lfs = [lfs[tuple[0]-1] for tuple in self.index_list]

        applier = SparkLFApplier(lfs)
        print(self.combined_df.rdd.collect())
        L = applier.apply(self.combined_df.rdd)
        label_model = LabelModel(cardinality=2)
        label_model.fit(L, n_epochs=500)
        # y_prob = label_model.predict_proba(L)[:, 1]
        y_prob = label_model.predict(L)
        pandas_df = self.combined_df.selectExpr("origin", "ST_AsText(ST_Transform(Geom,'epsg:3857','epsg:4326' )) as "
                                                          "Geom").toPandas()
        pandas_df['Label'] = y_prob
        pandas_df['long'] = pandas_df['Geom'].map(self.Long_from_geom)
        pandas_df['lat'] = pandas_df['Geom'].map(self.lat_from_geom)

        result = pandas_df.to_dict("records")
        return result

    def Long_from_geom(geom):
        a = geom.split("(")[-1].strip(')').split(" ")
        return float(a[0])

    def lat_from_geom(geom):
        a = geom.split("(")[-1].strip(')').split(" ")
        return float(a[1])
