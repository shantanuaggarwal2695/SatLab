EuroSAT dataset [ 21 , 22 ] contains 2 GB of 20K Sentinel-2 GeoTIFF images of 64×64px and 13 bands from 10 classes comprising 3 water body classes (sea, lakes, rivers) and 7 land body classes (resi-
dential, industrial, highways, forest, grassland, crops and pasture) in Europe. We aggregate these into two classes with water class containing 5.5K images and land class containing 14.5K images.

Link: https://github.com/phelber/EuroSAT

textual.csv: preprocessed images for EuroSAT with textural features



spatial.csv: preprocessed images for EuroSAT with spatial features using spatial operators like ST_Contains(), ST_Within(), ST_Distance(), ST_Overlap() and ST_Touches()
