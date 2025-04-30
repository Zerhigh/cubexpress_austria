import os
import time
import ee
import rasterio
import torch
import cubexpress
import shapely
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from typing import Tuple, List, Optional, Any, Dict, Hashable
from pathlib import Path
from tqdm import tqdm

from skimage import exposure
from rasterio.warp import reproject, Resampling
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import utils_histogram

# --------------------------------------------------
# Initialize the Earth Engine API
# --------------------------------------------------
try:
    ee.Initialize(project='ee-samuelsuperresolution')
except Exception as e:
    ee.Authenticate(auth_mode="notebook")
    ee.Initialize(project='ee-samuelsuperresolution')

# BASE = Path('/data/USERS/shollend/combined_download_nodata_test/')
#BASE = Path('/home/shollend/shares/users/master/dl2/combined_download')
BASE = Path('/data/USERS/shollend/combined_download')
#BASE = Path('./local_experiment/')

LABELS = (0, 40, 41, 42, 48, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 72, 83, 84, 87, 88, 92, 95, 96)
IMG_SHAPE = [512, 512]
UPSAMPLE = 4
DOWNSAMPLE = 0.25


class MatchedData:
    def __init__(self, id: str):
        self.id: str = id
        self.S2Batches: Dict[str, S2Batch] = {}
        self.best_tile: S2Batch | None = None

    def set_best_tile(self) -> None:
        if len(self.S2Batches) < 1:
            self.best_tile = None
        else:
            # get nodata percentage and correlation value
            s2_merged: Dict = {s2batch.s2_id: (s2batch.total_nodata_perc, s2batch.s2_corr)
                               for s2batch in self.S2Batches.values()}

            # Sort by decreasing nodata values (1-nodata) and correlation
            s2_sorted: Dict = dict(
                sorted(s2_merged.items(), key=lambda item: (1 - item[1][0], item[1][1]), reverse=True)
            )

            # return best key (first in sorted dict)
            best_s2_id: str = next(iter(s2_sorted))
            self.best_tile = self.S2Batches[best_s2_id]


class S2Batch:
    def __init__(self, s2_id: str):
        self.s2_id: str = s2_id
        self.s2_corr: float | None = None
        self.median_corr: float | None = None
        self.s2_nodata_perc: float | None = None
        self.total_nodata_perc: float | None = None
        self.lr_harm: np.ndarray | None = None
        self.hr_harm: np.ndarray | None = None
        self.lr_nodata: np.ndarray | None = None
        self.hr_nodata: np.ndarray | None = None
        self.s2_nodata: np.ndarray | None = None


def resample_mask_torch(arr: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Args:
        arr: (channel x width x height) array will be resampled to (channel x scale_factor*width x scale_factor*height)
        scale_factor: float, 0.25 for HR -> LR
                             4 for HR -> LR

    Returns: ret_arr

    """
    # nearest neigbor interpolation for masks allows strict adherence to lr mask
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=0)
    ret_arr = torch.nn.functional.interpolate(
        torch.from_numpy(arr).unsqueeze(0).float(),
        scale_factor=scale_factor,
        mode="nearest",
        #antialias=True
    ).squeeze().numpy()
    return (ret_arr.squeeze()).astype(bool)


def resample_torch(arr: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Args:
        arr: (channel x width x height) array will be resampled to (channel x scale_factor*width x scale_factor*height)
        scale_factor: float, 0.25 for HR -> LR
                             4 for HR -> LR

    Returns: ret_arr

    """
    ret_arr = torch.nn.functional.interpolate(
        torch.from_numpy(arr).unsqueeze(0),
        scale_factor=scale_factor,
        mode="bilinear",
        antialias=True
    ).squeeze().numpy()
    return ret_arr


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
        count = np.sum(mask == label)
        sats[f'dist_{label}'] = round(count / num_px, 5)
    return sats


def masked_match_histograms(image: np.ndarray,
                            reference: np.ndarray,
                            image_mask: np.ndarray,
                            reference_mask: np.ndarray,
                            fill_value=0):
    # adapted from https://gist.github.com/tayden/dcc83424ce55bfb970f60db3d4ddad18
    # to include masks for the input and reference image
    if image_mask.ndim < 3:
        image_mask = np.stack([image_mask]*4, axis=0)
    if reference_mask.ndim < 3:
        reference_mask = np.stack([reference_mask]*4, axis=0)
    masked_source_image = np.ma.array(image, mask=image_mask)
    masked_reference_image = np.ma.array(reference, mask=reference_mask)

    matched = np.ma.array(np.empty(image.shape, dtype=image.dtype),
                          mask=image_mask, fill_value=fill_value)

    # if the whole image is a mask == no values existing, return the masked array with 0s
    if matched.mask.all():
        #print(matched.filled())
        return matched.filled()
    else:
        for channel in range(masked_source_image.shape[0]):
            # print(np.unique(masked_source_image[channel].compressed()))
            # print(np.unique(masked_reference_image[channel].compressed()))
            print(masked_source_image[channel].compressed().shape)
            print(masked_reference_image[channel].compressed().shape)
            matched_channel = exposure.match_histograms(masked_source_image[channel].compressed(),
                                                        masked_reference_image[channel].compressed())

            # Re-insert masked background
            mask_ch = image_mask[channel]
            matched[channel][~mask_ch] = matched_channel.ravel()

        return matched.filled()


def to_squared_geopackage(data_: pd.DataFrame) -> gpd.GeoDataFrame:
    def to_point(row):
        return shapely.Point(row['lon'], row['lat'])

    def to_square(row, w=640):
        lons = [row['geometry'].x - 640,
                row['geometry'].x + 640,
                row['geometry'].x + 640,
                row['geometry'].x - 640,
                row['geometry'].x - 640, ]
        lats = [row['geometry'].y + 640,
                row['geometry'].y + 640,
                row['geometry'].y - 640,
                row['geometry'].y - 640,
                row['geometry'].y + 640, ]
        return shapely.Polygon(list(zip(lons, lats)))

    data_['geometry'] = data_.apply(to_point, axis=1)
    data = gpd.GeoDataFrame(data_, crs='EPSG:4326')
    data = data.to_crs(crs=31287)
    data['geometry'] = data.apply(to_square, axis=1)
    return data


def _process_batch(data: Tuple[Hashable, pd.DataFrame],
                   orthofoto_path: Path,
                   hr_compressed_mask_path: Path,
                   hr_orthofoto_path: Path,
                   hr_harm_path: Path,
                   lr_s2_path: Path,
                   lr_harm_path: Path, ) -> Dict:
    index, batch = data
    batch_statistics = {}
    try:
        ########### LOAD S2 TILES & ORTHOFOTOS ###########
        # first item should have same lat/lon as all others
        lat, lon, id = batch.iloc[0]['lat'], batch.iloc[0]['lon'], batch.iloc[0]['id']
        new_id = f"{id:05d}"
        matched_dataset = MatchedData(id=new_id)

        # Prepare the Sentinel-2 raster transform
        geotransform = cubexpress.lonlat2rt(
            lon=lon,
            lat=lat,
            edge_size=128,
            scale=10
        )
        gt = geotransform.geotransform
        s2_hr_trafo = rasterio.transform.Affine(gt['scaleX'] / UPSAMPLE, gt['shearX'], gt['translateX'],
                                                gt['shearY'], gt['scaleY'] / UPSAMPLE, gt['translateY'])

        # download and load all sentinel tiles
        download_sentinel2_samples(data=batch, geotransform=geotransform, output_path=BASE / 'tmp')
        batch_s2_paths = [Path(BASE / "tmp" / f"{row['s2_download_id']}.tif") for _, row in batch.iterrows()]

        # load all s2 tiles into memory (for all bands)
        s2_profile, s2_data = load_sentinel2_samples(path_list=batch_s2_paths)
        s2_profile.update(blockxsize=32, blockysize=32, nodata=0)

        # load corresponding orthofoto
        input_path = orthofoto_path / 'input' / f'input_{id}.tif'
        target_path = orthofoto_path / 'target' / f'target_{id}.tif'

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

        ########### MATCH FOR S2-TILES ###########

        # Binary masks for processing will be inverted to nodata=True in accordance with np.masked array Regulation
        # Binary masks saved and applied to images will saved as: nodata = False to allow binary
        # multiplication: image (4, 128, 128) * mask (128, 128)
        nodata_hr_mdata = np.all(mdata != 0, axis=0)  # (512, 512)
        nodata_hr_odata = np.all(odata != 0, axis=0)  # (512, 512)

        # All resampling will be conducted with Pytorch Pillow bilinear wrapper as it introduces the least errors
        # https://zuru.tech/blog/the-dangers-behind-image-resizing
        # methods for dimension filtering is included in resample_*-torch()
        nodata_lr_mdata = resample_mask_torch(arr=nodata_hr_mdata, scale_factor=0.25)  # (128, 128)
        nodata_lr_odata = resample_mask_torch(arr=nodata_hr_odata, scale_factor=0.25)  # (128, 128)

        # Harmonising and Matching
        for s2_name, s2_image in s2_data.items():
            s2_tile = S2Batch(s2_id=s2_name)

            # Sentinel-2 tile extraction: [4, 3, 2, 8] bands Required -> remapped to [3, 2, 1, 7] for R, G, B, NIR
            lr_s2_base = s2_image[[3, 2, 1, 7], :, :] / 10_000

            # create binary mask for filtering and homogenising all image data
            nodata_lr_s2 = np.all(lr_s2_base != 0, axis=0)  # (128, 128)

            # Rescale and apply the mask for the hr component
            nodata_lr = np.all(np.stack([nodata_lr_s2, nodata_lr_mdata, nodata_lr_odata], axis=0) != 0, axis=0)  # (128, 128)
            nodata_hr = resample_mask_torch(arr=nodata_lr, scale_factor=4)  # (512, 512)

            # Normalize images and calculate histogram matching
            hr_ortho_norm = odata / 255
            lr_s2_norm = lr_s2_base
            hr_harm_self = masked_match_histograms(image=hr_ortho_norm,
                                                   reference=lr_s2_norm,
                                                   image_mask=~nodata_hr,
                                                   reference_mask=~nodata_lr,
                                                   )

            # reduce with bilinear interpolation -> (4, 128, 128)
            lr_harm_self = resample_torch(arr=hr_harm_self, scale_factor=0.25)  # (4, 128, 128)

            # Compute block-wise correlation between LR and LRharm emitting NaN values
            corr = utils_histogram.fast_block_correlation(lr_s2_norm * nodata_lr_s2, lr_harm_self * nodata_lr,
                                                          none_value=0, block_size=16)

            # Report the 10th percentile of the correlation (low correlation) without nans
            low_corr = np.nanquantile(corr, 0.10)
            # weighted_corr_coefficient = low_corr * (1 - np.sign(low_corr) * np.sum(nodata_lr == 0)/(128*128))

            s2_tile.s2_corr = round(float(low_corr), 4)
            s2_tile.lr_harm = lr_harm_self
            s2_tile.hr_harm = hr_harm_self
            s2_tile.lr_nodata = nodata_lr
            s2_tile.hr_nodata = nodata_hr
            s2_tile.s2_nodata = nodata_lr_s2
            s2_tile.median_corr = np.nanmedian(corr)
            s2_tile.s2_nodata_perc = round(float(np.sum(nodata_lr_s2 == 0) / (128 * 128)), 5)
            s2_tile.total_nodata_perc = round(float(np.sum(nodata_lr == 0) / (128 * 128)), 5)

            matched_dataset.S2Batches[s2_tile.s2_id] = s2_tile

        # filter and set the best suited tile
        matched_dataset.set_best_tile()

        # apply shared masks
        masked_mdata = mdata * matched_dataset.best_tile.hr_nodata
        masked_odata = odata * matched_dataset.best_tile.hr_nodata

        ########### ADD METADATA ###########

        batch_statistics['low_corr'] = matched_dataset.best_tile.s2_corr
        batch_statistics['all_corrs'] = {tile.s2_id: tile.s2_corr for tile in matched_dataset.S2Batches.values()}
        batch_statistics['all_nodata_perc'] = {tile.s2_id: tile.total_nodata_perc for tile in matched_dataset.S2Batches.values()}
        batch_statistics['all_s2_ids'] = {r['s2_download_id']: r['s2_full_id'] for _, r in batch.iterrows()}
        batch_statistics['median_corr'] = matched_dataset.best_tile.median_corr
        batch_statistics['s2_available'] = True
        batch_statistics['nodata_total'] = matched_dataset.best_tile.total_nodata_perc
        batch_statistics['nodata_ortho'] = round(np.sum(nodata_hr_odata == 0) / (IMG_SHAPE[0] * IMG_SHAPE[1]), 5)
        batch_statistics['nodata_mask'] = round(np.sum(nodata_hr_mdata == 0) / (IMG_SHAPE[0] * IMG_SHAPE[1]), 5)
        batch_statistics['nodata_s2'] = matched_dataset.best_tile.s2_nodata_perc

        # add statistical info, except emtpy columns and statistics from the previous mask (before cropping)
        selected_row = batch.loc[batch["s2_download_id"] == matched_dataset.best_tile.s2_id]
        row_dict = selected_row.iloc[0].to_dict()
        bad_substrings = ['Unnamed', 'dist_', 's2_available']
        for k, v in row_dict.items():
            if not any(sub in k for sub in bad_substrings):
                batch_statistics[k] = v

        # Recalculate mask distribution statistics after masking with nodata
        mask_stats = calculate_mask_statistics(mask=masked_mdata)
        for k, v in mask_stats.items():
            batch_statistics[k] = v

        # add nodata flag based on the total no data count aggregated over all masks
        if batch_statistics['nodata_total'] > 0:
            batch_statistics['contains_nodata'] = True
        else:
            batch_statistics['contains_nodata'] = False

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
            dst.write((matched_dataset.best_tile.hr_harm * matched_dataset.best_tile.hr_nodata * 10_000).round().astype(rasterio.uint16))

        # Normal S2
        with rasterio.open(lr_s2_path / f'S2_{new_id}.tif', "w", **s2_profile) as dst:
            dst.write(s2_data[matched_dataset.best_tile.s2_id] * matched_dataset.best_tile.lr_nodata)

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
            dst.write((matched_dataset.best_tile.lr_harm * matched_dataset.best_tile.lr_nodata * 10_000).round().astype(rasterio.uint16))

        # remove unwanted sentinel images
        # for i, row in batch.iterrows():
        #     file = Path(BASE / "tmp" / f"{row['s2_download_id']}.tif")
        #     file.unlink()

        return batch_statistics
    except Exception as e:
        # In case of errors, append None
        print(f'Error at panda index {index}: {e}')
        return batch_statistics


if __name__ == '__main__':
    add_ = False
    if add_:
        table: pd.DataFrame = pd.read_csv('/home/shollend/shares/users/master/metadata/cubexpress/merged_ALL_S2_filter.csv')
        df_filtered: pd.DataFrame = add_more_metadata(input_table=table)
        df_filtered.to_csv('/home/shollend/shares/users/master/metadata/cubexpress/merged_ALL_S2_filter_wmeta_sample_150imgs.csv')
    else:
        df_filtered: pd.DataFrame = pd.read_csv(BASE / "/home/shollend/shares/users/master/metadata/cubexpress/merged_ALL_S2_filter_wmetadata.csv")
        #df_filtered: pd.DataFrame = pd.read_csv(Path("tables") / "merged_ALL_S2_filter_wmeta_sample_150imgs.csv")
        #df_filtered: pd.DataFrame = pd.read_csv(BASE / "sample_s2_wmeta.csv")

    # ortho path
    # inp_ortho_path = Path('/data/USERS/shollend/orthophoto/austria_full_allclasses_regridded')
    # inp_ortho_path = Path('C:/Users/PC/Coding/cubexpress_austria/local_experiment/local_orthofotos')

    # table path
    # /home/shollend/shares/users/master/metadata/cubexpress/merged_ALL_S2_filter.csv

    # base
    # /data/USERS/shollend/combined_download

    # generate folders
    #inp_ortho_path = Path('C:/Users/PC/Coding/cubexpress_austria/local_experiment/local_orthofotos')
    #inp_ortho_path = Path('/home/shollend/shares/users/master/dl2/orthofoto/austria_full_classes')
    inp_ortho_path = Path('/data/USERS/shollend/orthophoto/austria_full_classes_with_wine')

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

    # Aggregate S2-tiles to corresponding orthofoto
    df_batches = df_filtered.groupby("id")

    print('downloading sentinel2')
    tstart = time.time()

    # Create a multiprocessing list of tasks
    rows = [(idx, batch) for idx, batch in df_batches]

    res = []
    for row in tqdm(rows[:3]):
        res.append(_process_batch(data=row,
                                  orthofoto_path=inp_ortho_path,
                                  hr_compressed_mask_path=out_ortho_target,
                                  hr_orthofoto_path=out_ortho_input,
                                  hr_harm_path=out_ortho_input_harm,
                                  lr_s2_path=out_sentinel2,
                                  lr_harm_path=out_sentinel2_harm,
                                  ))

    # _process_batch_partial = partial(_process_batch,
    #                                  orthofoto_path = inp_ortho_path,
    #                                  hr_compressed_mask_path=out_ortho_target,
    #                                  hr_orthofoto_path=out_ortho_input,
    #                                  hr_harm_path=out_ortho_input_harm,
    #                                  lr_s2_path=out_sentinel2,
    #                                  lr_harm_path=out_sentinel2_harm,)
    #
    # # Run parallel processing
    # with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    #     res = list(tqdm(executor.map(_process_batch_partial, rows), total=len(rows)))

    out_df = pd.DataFrame.from_records(res)
    out_df.to_csv(BASE / 's2_ortho_download_data.csv')

    # out_gdf = to_squared_geopackage(data_=out_df)
    # out_gdf.to_file(str(BASE / 's2_ortho_download_data.gpkg'), driver='GPKG')

    tstop = time.time()
    print(f'Script took {round(tstop - tstart, 2)}s')

    for i, b in enumerate(res):
        if not b:
            print(f'Failed to process batch index/id: {i}')
