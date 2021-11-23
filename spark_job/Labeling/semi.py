from math import log2
from pyspark.ml.feature import VectorAssembler
import numpy as np


class SemiLabeling:
    def __init__(self, geo_df, text_df, spark):
        self.geo = geo_df
        self.text = text_df
        self.geo.createOrReplaceTempView("geo");
        self.text.createOrReplaceTempView("text")
        self.spark = spark
        self.combined_df = self.spark.sql(
            "select geo.origin, geo.Geom, geo.healthcare, geo.malls, geo.schools, geo.waste, geo.road, geo.forest, geo.residential, geo.power, geo.resort, geo.grasslands,text.glcm_contrast_Scaled, text.glcm_dissimilarity_Scaled, text.glcm_homogeneity_Scaled, text.glcm_energy_Scaled, text.glcm_correlation_Scaled, text.glcm_ASM_Scaled from text, geo where text.origin = geo.origin")
        self.rdd = self.combined_df.rdd
        self.data_count = self.combined_df.count()
        self.lfs = self.getweaklabels()
        self.theta = None

    def generate_threshold(self):
        clustering_prediction = self.getClustering().select("prediction").rdd.map(lambda x: x.prediction).collect()
        labeling_functions = self.getweaklabels()
        theta_distribution = []

        def kl(p, q):
            return sum(p[i] * log2(p[i] / q[i]) for i in range(len(p)))

        self.combined_df.persist()

        for i, lf in enumerate(labeling_functions):
            mle_theta = -1
            kl_min = 1
            for theta in np.arange(0.2, 1, 0.2):
                current_label = self.rdd.map(lambda x: lf(x, theta)).collect()
                p = [current_label.count(0) / (current_label.count(0) + current_label.count(1)),
                     current_label.count(1) / (current_label.count(0) + current_label.count(1))]
                q = [clustering_prediction.count(0) / (clustering_prediction.count(0) + clustering_prediction.count(1)),
                     clustering_prediction.count(1) / (clustering_prediction.count(0) + clustering_prediction.count(1))]
                kl_divergent = kl(p, q)
                #         print(kl_divergent)
                if kl_divergent < kl_min:
                    kl_min = kl_divergent
                    mle_theta = theta
            theta_distribution.append(mle_theta)
        return theta_distribution

    def generate_labels(self):
        self.theta = self.generate_threshold()

        def apply_lfs(x):
            labels = []
            for j, lf in enumerate(self.lfs):
                y = lf(x[0], self.theta[j])
                if y >= 0:
                    labels.append((x[1], j, y))
            return labels

        labels = self.rdd.zipWithIndex().map(apply_lfs).collect()
        print(labels)




    def getClustering(self):
        features = (
            "healthcare", "malls", "schools", "waste", "road", "forest", "residential", "power", "resort", "grasslands",
            "glcm_contrast_Scaled", "glcm_dissimilarity_Scaled", "glcm_homogeneity_Scaled", "glcm_energy_Scaled",
            "glcm_correlation_Scaled", "glcm_ASM_Scaled")
        # features = tuple(features)
        assembler = VectorAssembler(inputCols=features, outputCol="features")
        dataset = assembler.transform(self.combined_df)
        dataset.select("features").show(truncate=False)

        from pyspark.ml.evaluation import ClusteringEvaluator
        from pyspark.ml.clustering import KMeans

        kmeans = KMeans().setK(2).setSeed(1)
        model = kmeans.fit(dataset)

        predictions = model.transform(dataset)
        return predictions

    def getweaklabels(self):
        def lf1(x, theta):
            if x.healthcare < theta:
                return 1
            else:
                return 0

        def lf2(x, theta):
            if x.malls < theta:
                return 0
            else:
                return 1

        def lf3(x, theta):
            if x.schools < theta:
                return 0
            else:
                return 1

        def lf4(x, theta):
            if x.waste < theta:
                return 1
            else:
                return 0

        def lf5(x, theta):
            if x.forest < theta:
                return 1
            else:
                return 0

        def lf6(x, theta):
            if x.power < theta:
                return 0
            else:
                return 1

        def lf7(x, theta):
            if x.residential < theta:
                return 0
            else:
                return 1

        def lf8(x, theta):
            if x.road < theta:
                return 0
            else:
                return 1

        def lf9(x, theta):
            if x.resort < theta:
                return 0
            else:
                return 1

        def lf10(x, theta):
            if x.grasslands < theta:
                return 1
            else:
                return 0

        def lf11(x, theta):
            if x.glcm_contrast_Scaled < theta:
                return 1
            else:
                return 0

        def lf12(x, theta):
            if x.glcm_dissimilarity_Scaled < theta:
                return 1
            else:
                return 0

        def lf13(x, theta):
            if x.glcm_homogeneity_Scaled < theta:
                return 0
            else:
                return 1

        def lf14(x, theta):
            if x.glcm_energy_Scaled < theta:
                return 0
            else:
                return 1

        def lf15(x, theta):
            if x.glcm_correlation_Scaled < theta:
                return 0
            else:
                return 1

        def lf16(x, theta):
            if x.glcm_ASM_Scaled < theta:
                return 0
            else:
                return 1

        lfs = [lf1, lf2, lf3, lf4, lf5, lf6, lf7, lf8, lf9, lf10, lf11, lf12, lf13, lf14, lf15, lf16]
        return lfs
