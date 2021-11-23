from pyspark.ml.feature import VectorAssembler
from pyspark.sql.window import Window
from pyspark.sql.functions import lit,row_number,col
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import RandomForestClassifier



class Automatic:
    def __init__(self, geo_df, text_df, spark):
        self.geo = geo_df
        self.text = text_df
        self.geo.createOrReplaceTempView("geo");
        self.text.createOrReplaceTempView("text")
        self.spark = spark
        self.combined_df = self.spark.sql(
            "select geo.origin, geo.Geom, geo.healthcare, geo.malls, geo.schools, geo.waste, geo.road, geo.forest, geo.residential, geo.power, geo.resort, geo.grasslands,text.glcm_contrast_Scaled, text.glcm_dissimilarity_Scaled, text.glcm_homogeneity_Scaled, text.glcm_energy_Scaled, text.glcm_correlation_Scaled, text.glcm_ASM_Scaled from text, geo where text.origin = geo.origin")

    def parse(self, lines):
        block = []
        while lines:
            if lines[0].startswith('If'):
                bl = ' '.join(lines.pop(0).split()[1:]).replace('(', '').replace(')', '')
                block.append({'name': bl, 'children': self.parse(lines)})
                if lines[0].startswith('Else'):
                    be = ' '.join(lines.pop(0).split()[1:]).replace('(', '').replace(')', '')
                    block.append({'name': be, 'children': self.parse(lines)})
            elif not lines[0].startswith(('If', 'Else')):
                block2 = lines.pop(0)
                block.append({'name': block2})
            else:
                break
        return block

    def tree_json(self, tree):
        data = []
        for line in tree.splitlines():
            if line.strip():
                line = line.strip()
                data.append(line)
            else:
                break
            if not line: break
        res = []
        res.append({'name': 'Root', 'children': self.parse(data[1:])})
        return res


    def generate_rules(self, total_partitions):

        heuristics = []
        w = Window().partitionBy(lit('a')).orderBy(lit('a'))
        self.combined_df = self.combined_df.withColumn("row_num", row_number().over(w))

        partition = int(self.combined_df.count() / total_partitions)

        for i in range(0, total_partitions):
            df_partition = self.combined_df.filter(col("row_num").between(i * partition + 1, (i + 1) * partition))
            features = (
            "healthcare", "malls", "schools", "waste", "road", "forest", "residential", "power", "resort", "grasslands",
            "glcm_contrast_Scaled", "glcm_dissimilarity_Scaled", "glcm_homogeneity_Scaled", "glcm_energy_Scaled",
            "glcm_correlation_Scaled", "glcm_ASM_Scaled")
            assembler = VectorAssembler(inputCols=features, outputCol="features")
            dataset = assembler.transform(df_partition)
            kmeans = KMeans().setK(2).setSeed(1)
            model = kmeans.fit(dataset)
            predictions = model.transform(dataset)
            rf_df = predictions.selectExpr("features", "prediction as label")
            rf = RandomForestClassifier(numTrees=5, maxDepth=8, labelCol="label", seed=42)
            rf_model = rf.fit(rf_df)
            heuristics.append(self.tree_json(rf_model.toDebugString))

        return heuristics


