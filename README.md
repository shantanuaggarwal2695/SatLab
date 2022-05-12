In this work, we propose a novel, unsupervised framework ti-tled Satlab, to label satellite images, given a binary classifica-tion task. Existing models for satellite image classification such as 𝐷𝑒𝑒𝑝𝑆𝐴𝑇 and 𝐷𝑒𝑒𝑝𝑆𝐴𝑇 -𝑉 2 rely on deep-learning models that are label-hungry and require a significant amount of training data. Since manual curation of labels is expensive, we ensure that Satlab requires 𝑧𝑒𝑟𝑜 training labels. Satlab can work in conjunction with several generative and unsupervised ML models by allowing them to be seamlessly plugged into its architecture. We devise three operating modes for Satlab - 𝑚𝑎𝑛𝑢𝑎𝑙, 𝑠𝑒𝑚𝑖-𝑎𝑢𝑡𝑜𝑚𝑎𝑡𝑖𝑐 and 𝑓 𝑢𝑙𝑙𝑦-𝑎𝑢𝑡𝑜𝑚𝑎𝑡𝑖𝑐 which require varying levels of human intervention in creating the domain-specific labeling functions for each image that can be utilized by the candidate generative models such as 𝑆𝑛𝑜𝑟𝑘𝑒𝑙, as well as other unsupervised learners in Satlab. Unlike existing supervised learning baselines which only extract textural features from satellite images, we support the extraction of both textural and geospatial features in Satlab. We build Satlab on the top of 𝐴𝑝𝑎𝑐ℎ𝑒 𝑆𝑒𝑑𝑜𝑛𝑎 to leverage its rich set of spatial query processing operators for the extraction of geospatial features from satellite raster images. We evaluate Satlab on a slum classification dataset of 5M satellite images replicated from a seed set of 2K images and a two-class variant of a land/water classification dataset of 20K satellite images both captured by the Sentinel-2 satellite program. We empirically show that spatio-textural features enhance the classification F1-score by 23% for slum classification and 11.4% for land/water classification. We also show that Snorkel outperforms alternative generative and un- supervised candidate models that can be plugged into Satlab by 8.3% to 84% w.r.t. F1-score and 3× to 73× w.r.t. latency. Our 5-Fold Cross Validation (CV) experiments show that Satlab requires 0% labels and incurs up to 63% lower loading times and 3 orders of
magnitude lower learning latencies than the supervised learning baselines which utilize 80% labels to yield higher F1-scores.

This work contains two folders:

1. slum_labeling: This corresponds to first dataset related to slum labeling with the code and images

2. water_labeling: This folder corresponds to water-land labeling application with dataset from EuroSAT images.

We also show as user how to package every module in an .egg file for Pyspark and submit it to spark cluster using master node in slum_labeling folder which is very similar to running .jar file using spark

**These labeling functions can be changed with respect to different applications
**

Link to download Knowledge Base: https://download.geofabrik.de/
