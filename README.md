# Repository: cubexpress_austria

This repository relies on the [CubeXpress](https://github.com/JulioContrerasH/CubeXpress) package to download Sentinel-2 images from GEE. It is part of the master’s thesis: [Evaluating_Sentinel-2_Super-Resolution_Algorithms_for_Automated_Building_Delineation](https://github.com/Zerhigh/Evaluating_Sentinel-2_Super-Resolution_Algorithms_for_Automated_Building_Delineation)

This project contains scripts and utilities for downloading Sentinel-2 images from GEE which match corresponding orthophotos and cadastral masks. This allows the generation of evaluation datasets for Super-Resolution based building delineation.

The repository includes:

- `sample.py` — sample script highlighting the functionalities of the cubexpress package. this script is the foundation for the other processing scripts.   
- `collect_metadata.py` — reads in a sampling file with centroids of downloaded orthophoto/cadastral image patches and adds metadata for GEE querying
- `s2_download.py` — accesses the metadata annotated dataframe and downloades eight Sentinel-2 samples per centroid to select the bes-fitting one. selection criteria include high cloud-score (few clouds), few NoData values in any data source, temporal alignment, and high spectral correlation betweeen orthophoto and Sentinel-2 image.
- `utils_histogram.py` — helper functions for computing band-wise image correlations and apply cdf-matching.
---
