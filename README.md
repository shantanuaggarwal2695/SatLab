In this work, we propose a novel, unsupervised framework ti-tled Satlab, to label satellite images, given a binary classifica-tion task. Existing models for satellite image classification such as 𝐷𝑒𝑒𝑝𝑆𝐴𝑇 and 𝐷𝑒𝑒𝑝𝑆𝐴𝑇 -𝑉 2 rely on deep-learning models that are label-hungry and require a significant amount of training data. Since manual curation of labels is expensive, we ensure that Satlab requires 𝑧𝑒𝑟𝑜 training labels. Satlab can work in conjunction with several generative and unsupervised ML models by allowing them to be seamlessly plugged into its architecture. We devise three operating modes for Satlab - 𝑚𝑎𝑛𝑢𝑎𝑙, 𝑠𝑒𝑚𝑖-𝑎𝑢𝑡𝑜𝑚𝑎𝑡𝑖𝑐 and 𝑓 𝑢𝑙𝑙𝑦-𝑎𝑢𝑡𝑜𝑚𝑎𝑡𝑖𝑐 which require varying levels of human intervention in creating the domain-specific labeling functions for each image that can be utilized by the candidate generative models such as 𝑆𝑛𝑜𝑟𝑘𝑒𝑙, as well as other unsupervised learners in Satlab. Unlike existing supervised learning baselines which only extract textural features from satellite images, we support the extraction of both textural and geospatial features in Satlab. We build Satlab on the top of 𝐴𝑝𝑎𝑐ℎ𝑒 𝑆𝑒𝑑𝑜𝑛𝑎 to leverage its rich set of spatial query processing operators for the extraction of geospatial features from satellite raster images. We evaluate Satlab on a slum classification dataset of 5M satellite images replicated from a seed set of 2K images and a two-class variant of a land/water classification dataset of 20K satellite images both captured by the Sentinel-2 satellite program. We empirically show that spatio-textural features enhance the classification F1-score by 23% for slum classification and 11.4% for land/water classification. We also show that Snorkel outperforms alternative generative and un- supervised candidate models that can be plugged into Satlab by 8.3% to 84% w.r.t. F1-score and 3× to 73× w.r.t. latency. Our 5-Fold Cross Validation (CV) experiments show that Satlab requires 0% labels and incurs up to 63% lower loading times and 3 orders of
magnitude lower learning latencies than the supervised learning baselines which utilize 80% labels to yield higher F1-scores.

This work contains two folders:

1. slum_labeling: This corresponds to first dataset related to slum labeling with the code and images

2. water_labeling: This folder corresponds to water-land labeling application with dataset from EuroSAT images.

We also show as user how to package every module in an .egg file for Pyspark and submit it to spark cluster using master node in slum_labeling folder which is very similar to running .jar file using spark

**These labeling functions can be changed with respect to different applications
**

Link to download Knowledge Base: https://download.geofabrik.de/



**Slum Labeling**


gold_labels contains ground truth for 2k images for every image ID where 1 means a slum whereas a 0 denotes non-slum
Gold labels have been generated using some manual efforts based on some guidelines by UN as mentioned in the paper
The other folder contains 2k GeoTiff images in the format image_{ID} which is unique for every image and labeled accordingly.
The slum classification dataset contains 2K GeoTIFF images for two districts in Argentina: Buenos Aires and Córdoba (∼15M habitants). Each image comes from Sentinel-2A sensor with 32×32px and 4 bands (bands 2, 3, 4, 8A, 10 meter resolution) [ 8]. We manually generated the ground truth labels for the images using the labeling heuristics provided by the UN for slum classification [43 ]. For a scalable evaluation, we replicate the 2K images sized 18 MB into 5M images (69 GB). We partition the space of images into 4 quadrants and use a different replication factor for images from each quadrant. The replication factors were drawn from a Dirichlet distribution (𝛼 = 1) and we had 20%, 30%, 40% and 10% of the replicated images from the four quadrants. Every replicated image gets its ground truth label from the seed image. Our training and test sets are strictly non-overlapping during 5-Fold evaluation


**Water Labeling**


EuroSAT dataset [ 21 , 22 ] contains 2 GB of 20K Sentinel-2 GeoTIFF images of 64×64px and 13 bands from 10 classes comprising 3 water body classes (sea, lakes, rivers) and 7 land body classes (resi- dential, industrial, highways, forest, grassland, crops and pasture) in Europe. We aggregate these into two classes with water class containing 5.5K images and land class containing 14.5K images.

[21, 22]: https://github.com/phelber/EuroSAT

textual.csv: preprocessed images for EuroSAT with textural features

spatial.csv: preprocessed images for EuroSAT with spatial features using spatial operators like ST_Contains(), ST_Within(), ST_Distance(), ST_Overlap() and ST_Touches()
