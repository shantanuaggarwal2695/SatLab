Important Points:

1. gold_labels contains ground truth for 2k images for every image ID where 1 means a slum whereas a 0 denotes non-slum
2. Gold labels have been generated using some manual efforts based on some guidelines by UN as mentioned in the paper
3. The other folder contains 2k GeoTiff images in the format image_{ID} which is unique for every image and labeled accordingly.

Description:

The slum classification dataset contains 2K GeoTIFF images for two districts in Argentina: Buenos Aires and Córdoba (∼15M habitants). Each image comes from Sentinel-2A sensor with 32×32px and 4 bands (bands 2, 3, 4, 8A, 10 meter resolution) [ 8]. We manually generated the ground truth labels for the images using the labeling heuristics provided by the UN for slum classification [43 ]. For a scalable evaluation, we replicate the 2K images sized 18 MB into 5M images (69 GB). We partition the space of images into 4 quadrants and use a different replication factor for images from each quadrant. The replication factors were drawn from a Dirichlet distribution (𝛼 = 1) and we had 20%, 30%, 40% and 10% of the replicated images from the four quadrants. Every replicated image gets its ground truth label from the seed image. Our training and test sets are strictly non-overlapping during 5-Fold evaluation
