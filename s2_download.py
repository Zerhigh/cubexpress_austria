import ee
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Tuple, List, Optional
import utm
import os
from pathlib import Path
from tqdm import tqdm
import cubexpress
from typing import List, Optional
from utils_histogram import *

from multiprocessing import Pool
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor

from skimage.exposure import match_histograms
from scipy.ndimage import zoom

# ----------------------------------------------
# Initialize the Earth Engine API
# --------------------------------------------------
try:
    ee.Initialize()
except Exception:
    ee.Authenticate(auth_mode="notebook")
    ee.Initialize()

BASE = Path('')


def image_match(hr: np.array, lr: np.array) -> np.array:
    # Match histograms between HR and LR images
    hrharm = match_histograms(hr, lr, channel_axis=0)

    # Degrade HRharm to LRharm using bilinear interpolation
    lrharm = zoom(hrharm, (1, 0.25, 0.25), order=1)  # order=1 → Bilinear interpolation

    # Compute block-wise correlation between LR and LRharm
    kernel_size = 32
    corr = fast_block_correlation(lr, lrharm, block_size=kernel_size)
    return corr


def build_sentinel2_path(s2_id: str, sr_ids: List[str], other_ids: List[str]) -> str:
    """
    Return the appropriate Sentinel-2 image path based on ID availability.

    Args:
        other_ids: bla
        sr_ids: bla
        s2_id (str): The Sentinel-2 image ID (system:index).

    Returns:
        str: The full Sentinel-2 asset path.
    """
    if s2_id in sr_ids:
        return f"COPERNICUS/S2_SR_HARMONIZED/{s2_id}"
    elif s2_id in other_ids:
        return f"COPERNICUS/S2_HARMONIZED/{s2_id}"
    else:
        return f"UNKNOWN/{s2_id}"


def add_more_metadata(input_table: pd.DataFrame) -> pd.DataFrame:
    table: pd.DataFrame = input_table.copy()
    # You can change this because it's just a way to put new ids
    table["s2_download_id"] = [
        f"S2_U_{i:05d}" for i in range(len(table))
    ]

    # Filter out unique Sentinel-2 IDs
    filter_ids: List[str] = table["s2_id"].unique().tolist()

    # Check if IDs are in the SR (Surface Reflectance) collection
    ic_sr = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filter(ee.Filter.inList("system:index", filter_ids))
    )
    valid_sr_ids: List[str] = ic_sr.aggregate_array("system:index").getInfo()

    # Collect the remaining IDs not found in the SR collection
    valid_ids: List[str] = [
        item for item in filter_ids if item not in valid_sr_ids
    ]

    # Build full Sentinel-2 paths in the table
    table["s2_full_id"] = table["s2_id"].apply(build_sentinel2_path, sr_ids=valid_sr_ids, other_ids=valid_ids)

    # Filter the dataframe for SR images only
    df_filtered: pd.DataFrame = table[
        table["s2_full_id"].str.startswith("COPERNICUS/S2_SR_HARMONIZED/")
    ].copy()

    return df_filtered


def _process_batch(data: Tuple[int, pd.DataFrame]) -> Tuple[int, bool]:
    index, batch = data
    try:
        # first item should have same lat/lon as all others
        lat, lon = batch.iloc[0]['lat'], batch.iloc[0]['lon']
        # Prepare the raster transform
        geotransform = cubexpress.lonlat2rt(
            lon=lon,
            lat=lat,
            edge_size=128,
            scale=10
        )

        requests = []
        for i, row in batch.iterrows():
            # Build a single Request
            requests.append(cubexpress.Request(id=row["s2_download_id"],
                                               raster_transform=geotransform,
                                               bands=[
                                                   "B1", "B2", "B3", "B4", "B5",
                                                   "B6", "B7", "B8", "B8A", "B9",
                                                   "B11", "B12"
                                               ],
                                               image=row["s2_full_id"]
                                               ))

        # Create a RequestSet (can hold multiple requests)
        cube_requests = cubexpress.RequestSet(requestset=requests)

        # Fetch the data cube
        cubexpress.getcube(
            request=cube_requests,
            output_path=BASE / "output",  # directory for output
            nworkers=4,  # parallel workers
            max_deep_level=5  # maximum concurrency with Earth Engine
        )

        # load all sentinel images

        # load corresponding orthofoto

        # resample/crop/transform

        # histogram matching

        # save values

        # remove unwanted sentinel images

        return index, True
    except Exception as e:
        # In case of errors, append None
        print(f'Error: {e}')
        return index, False


if __name__ == '__main__':
    add_ = False
    if add_:
        table: pd.DataFrame = pd.read_csv(BASE / "tables" / "ccs_090_ALL_S2_filter_sample200.csv")
        df_filtered: pd.DataFrame = add_more_metadata(input_table=table)
        df_filtered.to_csv(BASE / 'tables' / 'ccs_090_ALL_S2_filter_sample200_withmetadata.csv')
    else:
        df_filtered: pd.DataFrame = pd.read_csv(BASE / "tables" / "ccs_090_ALL_S2_filter_sample200_withmetadata.csv")

    # Download with cubexpress
    df_batches = df_filtered[:20].groupby("id")

    print('downloading sentinel2')
    # Create a multiprocessing pool
    rows = [(idx, batch) for idx, batch in df_batches]

    #res = [_process_batch(row) for row in rows]

    # Use ProcessPoolExecutor instead of multiprocessing.Pool
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        res = list(tqdm(executor.map(_process_batch, rows), total=len(df_batches)))

    for i, b in res:
        if not b:
            print(f'Failed to process batch index/id: {i}')