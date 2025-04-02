import ee
import numpy as np
import pandas as pd
from osgeo import gdal  # Import gdal before rasterio
import geopandas as gpd
from typing import Tuple, List, Optional, Any, Dict, Hashable
import rasterio
from skimage.transform import resize
import utm
import os
from pathlib import Path
import shutil
from tqdm import tqdm
import cubexpress
from typing import List, Optional
import utils_histogram
from functools import partial

import skimage

from rasterio.warp import reproject, Resampling
from shapely.geometry import box

from multiprocessing import Pool
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor

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
UPSAMPLE = 4
DOWNSAMPLE = 0.25


# austria = gpd.read_file('C:/Users/PC/Desktop/TU/Master/MasterThesis/data/metadata/oesterreich_border/oesterreich.shp')
# austria32 = austria.to_crs('32632')
# austria33 = austria.to_crs('32633')
# GEOMS = {32: austria32.loc[[0], 'geometry'].values[0], 33: austria33.loc[[0], 'geometry'].values[0]}


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


def load_sentinel2_samples(path_list: List[Path]) -> Tuple[Any, Dict[str, np.ndarray]]:
    # loads a number of sentinel2 samples and stores their arrays in a dict with its name_id as a key, the profile of
    # the last sample is also provided
    data = {}
    for file in path_list:
        name = file.stem
        with rasterio.open(file, 'r') as src:
            data[name] = src.read()
            profile = src.profile

    # verify profiles are the same
    # just return the last one
    return profile, data


def transform_ortho(ortho_fp: str | Path,
                    s2_crs: str,
                    s2_transform: rasterio.transform.Affine,
                    s2_width: int,
                    s2_height: int) -> Tuple[np.ndarray, Dict]:  # | Tuple[np.ndarray, Dict]:
    # open orthofoto image tile
    with rasterio.open(ortho_fp) as osrc:
        src_data = osrc.read()
        src_crs = osrc.crs

        # define empty array and reproject data into it
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
        return dst_data, profile


def download_sentinel2_samples(data: pd.DataFrame, geotransform: Any, output_path: Path) -> None:
    requests = []
    # download only the few relevant bands for matching!
    for i, row in data.iterrows():
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
        output_path=output_path,
        nworkers=4,
        max_deep_level=5
    )
    return


def calculate_mask_statistics(mask: np.ndarray) -> Dict[str, float]:
    sats = {}
    num_px = IMG_SHAPE[0] * IMG_SHAPE[1]

    for label in LABELS:
        count = np.count_nonzero(mask == label)
        sats[f'dist_{label}'] = round(count / num_px, 3)
    return sats


def _process_batch(data: Tuple[Hashable, pd.DataFrame],
                   hr_compressed_mask_path: Path,
                   hr_orthofoto_path: Path,
                   hr_harm_path: Path,
                   lr_s2_path: Path,
                   lr_harm_path: Path) -> Dict:

    index, batch = data
    batch_statistics = {}
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
        s2_hr_trafo = rasterio.transform.Affine(gt['scaleX'] / UPSAMPLE, gt['shearX'], gt['translateX'],
                                                gt['shearY'], gt['scaleY'] / UPSAMPLE, gt['translateY'])

        # download and load all sentinel images into memory (for all bands)
        download_sentinel2_samples(data=batch, geotransform=geotransform, output_path=BASE / 'tmp')
        batch_s2_paths = [Path(BASE / "tmp" / f"{row['s2_download_id']}.tif") for _, row in batch.iterrows()]
        s2_profile, s2_data = load_sentinel2_samples(path_list=batch_s2_paths)
        s2_profile.update(
            blockxsize=32,
            blockysize=32,
            nodata=0,
            # compress="zstd",
            # zstd_level=13,
            # interleave="band",
            # discard_lsb=2
        )

        # load corresponding orthofoto
        input_path = BASE / 'local_orthofotos' / 'input' / f'input_{id}.tif'
        target_path = BASE / 'local_orthofotos' / 'target' / f'target_{id}.tif'

        # transform and resample the orthofoto images and masks
        mdata, mprofile = transform_ortho(ortho_fp=target_path,
                                          s2_crs=s2_profile['crs'],
                                          s2_transform=s2_hr_trafo,
                                          s2_width=IMG_SHAPE[0],
                                          s2_height=IMG_SHAPE[1],
                                          )

        odata, oprofile = transform_ortho(ortho_fp=input_path,
                                          s2_crs=s2_profile['crs'],
                                          s2_transform=s2_hr_trafo,
                                          s2_width=IMG_SHAPE[0],
                                          s2_height=IMG_SHAPE[1],
                                          )

        # ADD S2 NODATA MASK - Currently ignored on purpose
        # create boolean mask for filtering and homogenising all image data
        nodata_mask = np.all(np.concatenate([mdata, odata], axis=0) != 0, axis=0)
        masked_mdata = mdata * nodata_mask
        masked_odata = odata * nodata_mask

        # Recalculate mask distribution statistics after masking with nodata
        mask_stats = calculate_mask_statistics(mask=masked_mdata)
        for k, v in mask_stats.items():
            batch_statistics[k] = v

        # Degrade HR to LR using bilinear interpolation AND normalize (order=1 → Bilinear interpolation)

        lr_nodata_mask = resize(nodata_mask, (128, 128), anti_aliasing=False)
        hr_ortho_norm = masked_odata / 255
        #hr_ortho_norm = resize(masked_odata / 255, (4, 128, 128), anti_aliasing=False)

        #lr_ortho = resize(masked_odata / 255, (4, 128, 128), anti_aliasing=False)
        #lr_ortho = zoom(masked_odata / 255, (1, DOWNSAMPLE, DOWNSAMPLE), order=1)
        #lr_ortho = zoom(masked_odata / 255, (1, 0.25, 0.25), order=1)

        normalized_nodata_value = 0 / 255

        # Harmonising and Matching
        s2_corrs, lr_harms, hr_harms = {}, {}, {}
        for s2_name, s2_image in s2_data.items():
            # apply mask and normalize
            # [4, 3, 2, 8] bands Required -> remapped to [3, 2, 1, 7]
            lr_s2 = (s2_image[[3, 2, 1, 7], :, :] * lr_nodata_mask) / 10_000

            # histogram matching -> output 4, 512, 512
            hr_harm_self = utils_histogram.match_histograms(image=hr_ortho_norm, reference=lr_s2, channel_axis=0, ignore_none=True, none_value=normalized_nodata_value)

            # reduce with bilinear interpolation -> output 4, 128, 128
            lr_harm_self = resize(hr_harm_self, (4, 128, 128), anti_aliasing=False)

            # Compute block-wise correlation between LR and LRharm
            kernel_size = 32
            #corr_self = utils_histogram.fast_block_correlation(lr_s2, lr_harm_self, block_size=kernel_size, none_value=normalized_nodata_value)
            corr_self = utils_histogram.own_bandwise_correlation(lr_s2, lr_harm_self, none_value=normalized_nodata_value)

            # Report the 10th percentile of the correlation (low correlation) without nans
            low_cor_self = np.nanquantile(corr_self, 0.10)

            s2_corrs[s2_name] = round(low_cor_self, 4)
            lr_harms[s2_name] = lr_harm_self
            hr_harms[s2_name] = hr_harm_self

            # test with official changed method (including no data)
            # hr_harm = utils_histogram.real_match_histograms(image=hr_ortho_norm, reference=lr_s2, channel_axis=0)
            # lr_harm = resize(hr_harm, (4, 128, 128), anti_aliasing=False)
            # corr = utils_histogram.fast_block_correlation(lr_s2, lr_harm, block_size=kernel_size)
            # low_cor = np.nanquantile(corr, 0.10)

        # Best fitting sentinel2 sample for the current orthophoto
        best_s2_key = max(s2_corrs, key=s2_corrs.get)

        # add statistical info, except emtpy columns and statistics from the previous mask (before cropping)
        selected_row = batch.loc[batch["s2_download_id"] == best_s2_key]
        row_dict = selected_row.iloc[0].to_dict()
        for k, v in row_dict.items():
            if 'Unnamed' not in k or 'dist_' not in k:
                batch_statistics[k] = v

        batch_statistics['low_corr'] = s2_corrs[best_s2_key]

        ########### WRITE FILES ###########
        batch_statistics['hr_mask_path'] = hr_compressed_mask_path / f'HR_mask_{new_id}.tif'
        batch_statistics['hr_othofoto_path'] = hr_orthofoto_path / f'HR_ortho_{new_id}.tif'
        batch_statistics['hr_harm_path'] = hr_harm_path / f'HR_ortho_{new_id}.tif'
        batch_statistics['lr_s2_path'] = lr_s2_path / f'S2_{new_id}.tif'
        batch_statistics['lr_harm_path'] = lr_harm_path / f'S2_{new_id}.tif'

        # Compressed HR-Mask
        mprofile.update(
            dtype=rasterio.uint8,
            compress="zstd",
            zstd_level=13,
            interleave="band",
            tiled=True,
            blockxsize=128,
            blockysize=128,
            nodata=0,
        )
        with rasterio.open(hr_compressed_mask_path / f'HR_mask_{new_id}.tif', mode='w+', **mprofile) as dst:
            dst.write(masked_mdata)

        # Normal HR-Orthofoto
        with rasterio.open(hr_orthofoto_path / f'HR_ortho_{new_id}.tif', mode='w+', **oprofile) as dst:
            dst.write(masked_odata)

        # Harmonized HR-Orthofoto
        harm_hr_profile = oprofile.copy()
        harm_hr_profile.update(
            dtype=rasterio.uint16,
            compress="zstd",
            zstd_level=13,
            interleave="band",
            tiled=True,
            blockxsize=128,
            blockysize=128,
            discard_lsb=2,
            nodata=0,
        )

        with rasterio.open(hr_harm_path / f'HR_ortho_{new_id}.tif', "w", **harm_hr_profile) as dst:
            dst.write((hr_harms[best_s2_key] * 10_000).round().astype(rasterio.uint16))

        # Normal S2 - NODAtA update?
        # file = Path(BASE / "tmp" / f"{best_s2_key}.tif")
        # file.rename(lr_s2_path / f'S2_{new_id}.tif')
        with rasterio.open(lr_s2_path / f'S2_{new_id}.tif', "w", **s2_profile) as dst:
            lr_save_data = s2_data[best_s2_key] * lr_nodata_mask
            dst.write(lr_save_data)

        # Harmonized S2
        harm_lr_profile = s2_profile.copy()
        harm_lr_profile.update(
            dtype=rasterio.uint16,
            compress="zstd",
            count=4,
            zstd_level=13,
            interleave="band",
            tiled=True,
            discard_lsb=2
        )

        with rasterio.open(lr_harm_path / f'S2_{new_id}.tif', "w", **harm_lr_profile) as dst:
            dst.write((lr_harms[best_s2_key] * 10_000).round().astype(rasterio.uint16))

        # remove unwanted sentinel images
        for i, row in batch.iterrows():
            file = Path(BASE / "tmp" / f"{row['s2_download_id']}.tif")
            file.unlink()

        batch_statistics['s2_available'] = True

        return batch_statistics
    except Exception as e:
        # In case of errors, append None
        print(f'Error at panda index {index}: {e}')
        return batch_statistics


if __name__ == '__main__':
    add_ = False
    if add_:
        table: pd.DataFrame = pd.read_csv(BASE / "sample_s2.csv")
        df_filtered: pd.DataFrame = add_more_metadata(input_table=table)
        df_filtered.to_csv(BASE / "sample_s2_wmeta.csv")
    else:
        # df_filtered: pd.DataFrame = pd.read_csv(BASE / "tables" / "ccs_090_ALL_S2_filter_sample200_withmetadata.csv")
        df_filtered: pd.DataFrame = pd.read_csv(BASE / "sample_s2_wmeta.csv")

    # get statelog
    statelog: pd.DataFrame = pd.read_csv(
        "C:/Users/PC/Desktop/TU/Master/MasterThesis/data/metadata/statelogs/austria_full_allclasses_regridded/statelog.csv")

    # generate folders
    out_ortho_target = BASE / 'output' / 'hr_mask'
    out_ortho_input = BASE / 'output' / 'hr_orthofoto'
    out_ortho_input_harm = BASE / 'output' / 'hr_harm_orthofoto'
    out_sentinel2 = BASE / 'output' / 'lr_s2'
    out_sentinel2_harm = BASE / 'output' / 'lr_harm_s2'
    tmp = BASE / 'tmp'

    out_ortho_input.mkdir(parents=True, exist_ok=True)
    out_ortho_target.mkdir(parents=True, exist_ok=True)
    out_sentinel2.mkdir(parents=True, exist_ok=True)
    out_ortho_input_harm.mkdir(parents=True, exist_ok=True)
    out_sentinel2_harm.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)

    # Download with cubexpress
    df_batches = df_filtered.groupby("id")

    print('downloading sentinel2')
    # Create a multiprocessing pool
    rows = [(idx, batch) for idx, batch in df_batches]

    # res = [_process_batch(row) for row in rows]
    # res = []
    # for row in rows[:3]:
    #     res.append(_process_batch(data=row,
    #                               hr_compressed_mask_path=out_ortho_target,
    #                               hr_orthofoto_path=out_ortho_input,
    #                               hr_harm_path=out_ortho_input_harm,
    #                               lr_s2_path=out_sentinel2,
    #                               lr_harm_path=out_sentinel2_harm
    #                               ))

    _process_batch_partial = partial(_process_batch,
                                     hr_compressed_mask_path=out_ortho_target,
                                     hr_orthofoto_path=out_ortho_input,
                                     hr_harm_path=out_ortho_input_harm,
                                     lr_s2_path=out_sentinel2,
                                     lr_harm_path=out_sentinel2_harm)

    # Run parallel processing
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        res = list(tqdm(executor.map(_process_batch_partial, rows[:3]), total=len(rows)))

    out_df = pd.DataFrame.from_records(res)
    out_df.to_csv(BASE / 'test_process.csv')

    for i, b in enumerate(res):
        if not b:
            print(f'Failed to process batch index/id: {i}')
