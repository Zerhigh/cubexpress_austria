import ee
import numpy as np
import pandas as pd
from osgeo import gdal # Import gdal before rasterio
import geopandas as gpd
from typing import Tuple, List, Optional, Any, Dict
import rasterio
import utm
import os
from pathlib import Path
import shutil
from tqdm import tqdm
import cubexpress
from typing import List, Optional
from utils_histogram import *

from rasterio.warp import reproject, Resampling
from shapely.geometry import box

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

BASE = Path('local_experiment')
LABELS = (0, 40, 41, 42, 48, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 72, 83, 84, 87, 88, 92, 95, 96)
IMG_SHAPE = [512, 512]
# austria = gpd.read_file('C:/Users/PC/Desktop/TU/Master/MasterThesis/data/metadata/oesterreich_border/oesterreich.shp')
# austria32 = austria.to_crs('32632')
# austria33 = austria.to_crs('32633')
# GEOMS = {32: austria32.loc[[0], 'geometry'].values[0], 33: austria33.loc[[0], 'geometry'].values[0]}


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


def load_s2_tiles(path_list: List[Path]) -> Tuple[Any, Dict[str, Any]]:
    profiles = []
    data = {}
    for file in path_list:
        name = file.stem
        with rasterio.open(file, 'r') as src:
            profiles.append(src.profile)
            data[name] = src.read()

    # verify profiles are the same
    return profiles[0], data


def transform_ortho(ortho_fp: str | Path,
                    s2_crs: str,
                    s2_transform: rasterio.transform.Affine,
                    s2_width: int,
                    s2_height: int) -> np.ndarray | Tuple[np.ndarray, Dict]:

    # open orthofoto image tile
    with rasterio.open(ortho_fp) as osrc:
        src_data = osrc.read()
        src_crs = osrc.crs

        # define empt arrar and reproject data into it
        dst_data = np.zeros(shape=(osrc.count, s2_width, s2_height), dtype='uint8')

        a, _ = reproject(
            source=src_data,
            destination=dst_data,
            src_transform=osrc.transform,
            src_crs=src_crs,
            dst_transform=s2_transform,
            dst_crs=s2_crs,
            resampling=Resampling.nearest
        )

        profile = osrc.profile
        profile.update(transform=s2_transform, crs=s2_crs, width=s2_width, height=s2_height)

        # write data to file
        # with rasterio.open(ortho_op, 'w+', **profile) as dst:
        #     dst.write(dst_data)

        # if its the mask, recalculate the statitics
        sats = {}
        if osrc.count == 1:
            # update metadata count etc
            num_px = s2_width * s2_height
            # counts cannot be set as its rasterized!
            instance_counts = None #gdf['label'].value_counts().to_dict()

            for label in LABELS:
                count = np.count_nonzero(dst_data == label)

                sats[f'dist_{label}'] = round(count / num_px, 3)
                sats[f'count_{label}'] = instance_counts

        return dst_data, profile, sats


def _process_batch(data: Tuple[int, pd.DataFrame]) -> Tuple[int, bool]:
    index, batch = data
    output_panda = []
    try:
        # first item should have same lat/lon as all others
        lat, lon, id = batch.iloc[0]['lat'], batch.iloc[0]['lon'], batch.iloc[0]['id']
        new_id = f"{id:05d}"
        # Prepare the raster transform
        geotransform = cubexpress.lonlat2rt(
            lon=lon,
            lat=lat,
            edge_size=128,
            scale=10
        )
        gt = geotransform.geotransform
        new_x_scale = gt['scaleX'] / 4
        new_y_scale = gt['scaleY'] / 4

        s2_hr_trafo = rasterio.transform.Affine(new_x_scale, gt['shearX'], gt['translateX'],
                                                gt['shearY'], new_y_scale, gt['translateY'])

        requests = []
        for i, row in batch.iterrows():
            # Build a single Request
            # download only the few relevant bands for matching!
            requests.append(cubexpress.Request(id=row["s2_download_id"],
                                               raster_transform=geotransform,
                                               bands=["B2", "B3", "B4", "B8"],
                                               image=row["s2_full_id"]
                                               ))
            # requests.append(cubexpress.Request(id=row["s2_download_id"],
            #                                    raster_transform=geotransform,
            #                                    bands=[
            #                                        "B1", "B2", "B3", "B4", "B5",
            #                                        "B6", "B7", "B8", "B8A", "B9",
            #                                        "B11", "B12"
            #                                    ],
            #                                    image=row["s2_full_id"]
            #                                    ))

        # Create a RequestSet (can hold multiple requests)
        cube_requests = cubexpress.RequestSet(requestset=requests)

        #Fetch the data cube
        cubexpress.getcube(
            request=cube_requests,
            output_path=BASE / "tmp",  # directory for output
            nworkers=4,  # parallel workers
            max_deep_level=5  # maximum concurrency with Earth Engine
        )

        # load all sentinel images
        cmb_s2_paths = [Path(BASE / "tmp" / f"{row['s2_download_id']}.tif") for _, row in batch.iterrows()]
        s2_profile, s2_data = load_s2_tiles(path_list=cmb_s2_paths)

        # load corresponding orthofoto
        input_path = BASE / 'local_orthofotos' / 'input' / f'input_{id}.tif'
        target_path = BASE / 'local_orthofotos' / 'target' / f'target_{id}.tif'

        tinput_path = BASE / 'output' / 'hr_orthofoto'
        ttarget_path = BASE / 'output' / 'hr_mask'

        # transform and resample the orthofoto images and masks
        mdata, mprofile, mask_stats = transform_ortho(ortho_fp=target_path,
                                s2_crs=s2_profile['crs'],
                                s2_transform=s2_hr_trafo,
                                s2_width=IMG_SHAPE[0], #s2_profile['width'],
                                s2_height=IMG_SHAPE[1], #s2_profile['height'],
                                )

        # reapply the newly calculated mask statistics
        output_panda.append(mask_stats)

        odata, oprofile, _ = transform_ortho(ortho_fp=input_path,
                            s2_crs=s2_profile['crs'],
                            s2_transform=s2_hr_trafo,
                            s2_width=IMG_SHAPE[0],  # s2_profile['width'],
                            s2_height=IMG_SHAPE[1],  # s2_profile['height'],
                            )

        # create boolean mask for filtering and homogenising all image data
        nodata_mask = np.all(np.concatenate([mdata, odata], axis=0) != 0, axis=0)

        masked_mdata = mdata * nodata_mask
        masked_odata = odata * nodata_mask

        # apply mask and save to file
        with rasterio.open(ttarget_path / f'HR_mask_{new_id}.tif', mode='w+', **mprofile) as dst:
            dst.write(masked_mdata)

        with rasterio.open(tinput_path / f'HR_ortho_{new_id}.tif', mode='w+', **oprofile) as dst:
            dst.write(masked_odata)

        # Degrade HR to LR using bilinear interpolation AND normalize
        lr_nodata = zoom(nodata_mask, (0.25, 0.25), order=1)  # order=1 → Bilinear interpolation
        lr_ortho = zoom(masked_odata / 255, (1, 0.25, 0.25), order=1)

        s2_corrs = {}
        # adjust sentinel 2 images with nodata mask if necessary
        for s2_name, s2_image in s2_data.items():
            # apply mask and normalize
            lr_s2 = (s2_image * lr_nodata) / 10_000

            # histogram matching
            lr_harm = match_histograms(lr_ortho, lr_s2, channel_axis=0)

            # Compute block-wise correlation between LR and LRharm
            kernel_size = 32
            corr = fast_block_correlation(lr_s2, lr_harm, block_size=kernel_size)

            # Report the 10th percentile of the correlation (low correlation) without nans
            low_cor = np.nanquantile(corr, 0.10)
            s2_corrs[s2_name] = low_cor

        best_s2_key = max(s2_corrs, key=s2_corrs.get)
        output_panda.append({'low_corr': s2_corrs[best_s2_key]})

        # remove unwanted sentinel images
        for i, row in batch.iterrows():
            file = Path(BASE / "tmp" / f"{row['s2_download_id']}.tif")
            # delete file
            if row["s2_download_id"] != best_s2_key:
                file.unlink()
            else:
                request = cubexpress.Request(id=row["s2_download_id"],
                                               raster_transform=geotransform,
                                               bands=[
                                                   "B1", "B2", "B3", "B4", "B5",
                                                   "B6", "B7", "B8", "B8A", "B9",
                                                   "B11", "B12"
                                               ],
                                               image=row["s2_full_id"]
                                               )

                # Create a RequestSet (can hold multiple requests)
                cube_requests = cubexpress.RequestSet(requestset=[request])

                # Fetch the data cube
                cubexpress.getcube(
                    request=cube_requests,
                    output_path=BASE / "tmp",  # directory for output
                    nworkers=4,  # parallel workers
                    max_deep_level=5  # maximum concurrency with Earth Engine
                )

                # apply mask and save to file
                # with rasterio.open(file, mode='r+') as dst:
                #     data = dst.read() * lr_nodata
                #     dst.write(data)

                file.rename(BASE / 'output' / 'lr_s2' / f'S2_{new_id}.tif')

        return index, True
    except Exception as e:
        # In case of errors, append None
        print(f'Error: {e}')
        return index, False


if __name__ == '__main__':
    add_ = False
    if add_:
        table: pd.DataFrame = pd.read_csv(BASE / "sample_s2.csv")
        df_filtered: pd.DataFrame = add_more_metadata(input_table=table)
        df_filtered.to_csv(BASE / "sample_s2_wmeta.csv")
    else:
        #df_filtered: pd.DataFrame = pd.read_csv(BASE / "tables" / "ccs_090_ALL_S2_filter_sample200_withmetadata.csv")
        df_filtered: pd.DataFrame = pd.read_csv(BASE / "sample_s2_wmeta.csv")

    # get statelog
    statelog: pd.DataFrame = pd.read_csv("C:/Users/PC/Desktop/TU/Master/MasterThesis/data/metadata/statelogs/austria_full_allclasses_regridded/statelog.csv")

    # generate folders
    out_ortho_input = BASE / 'output' / 'hr_orthofoto'
    out_ortho_target = BASE / 'output' / 'hr_mask'
    out_sentinel2 = BASE / 'output' / 'lr_s2'
    tmp = BASE / 'tmp'

    out_ortho_input.mkdir(parents=True, exist_ok=True)
    out_ortho_target.mkdir(parents=True, exist_ok=True)
    out_sentinel2.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)

    # Download with cubexpress
    df_batches = df_filtered.groupby("id")

    print('downloading sentinel2')
    # Create a multiprocessing pool
    rows = [(idx, batch) for idx, batch in df_batches]

    #res = [_process_batch(row) for row in rows]
    res = []
    for row in rows:
        res.append(_process_batch(row))
        break

    # Use ProcessPoolExecutor instead of multiprocessing.Pool
    # with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    #     res = list(tqdm(executor.map(_process_batch, rows), total=len(df_batches)))

    for i, b in res:
        if not b:
            print(f'Failed to process batch index/id: {i}')