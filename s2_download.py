import os
# os.environ['CONDA_PREFIX'] = 'C:/Users/PC/anaconda3/envs/cubexpress_austria2/Library/share/gdal/'
# os.environ['GDAL_DATA'] = os.environ['CONDA_PREFIX'] + '/Library/share/gdal'
# os.environ['PROJ_LIB'] = os.environ['CONDA_PREFIX'] + '/Library/share'

import time
import ee
import rasterio
import torch
import cubexpress
import numpy as np
import pandas as pd

from typing import Tuple, List, Optional, Any, Dict, Hashable
from pathlib import Path
from tqdm import tqdm

from skimage.exposure import histogram_matching
from rasterio.warp import reproject, Resampling
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import utils_histogram

# --------------------------------------------------
# Initialize the Earth Engine API
# --------------------------------------------------
try:
    ee.Initialize(project='ee-samuelsuperresolution')
except Exception:
    ee.Authenticate(auth_mode="notebook")
    ee.Initialize(project='ee-samuelsuperresolution')

#BASE = Path('/data/USERS/shollend/combined_download_nodata_test/')
#BASE = Path('/home/shollend/shares/users/master/dl2/combined_download')
BASE = Path('./local_experiment/')

LABELS = (0, 40, 41, 42, 48, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 72, 83, 84, 87, 88, 92, 95, 96)
IMG_SHAPE = [512, 512]
UPSAMPLE = 4
DOWNSAMPLE = 0.25


def resample_mask_torch(arr: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Args:
        arr: (channel x width x height) array will be resampled to (channel x scale_factor*width x scale_factor*height)
        scale_factor: float, 0.25 for HR -> LR
                             4 for HR -> LR

    Returns: ret_arr

    """
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=0)
    ret_arr = torch.nn.functional.interpolate(
        torch.from_numpy(arr).unsqueeze(0).float(),
        scale_factor=scale_factor,
        mode="bilinear",
        antialias=True
    ).squeeze().numpy()
    return (ret_arr.squeeze() > 0.5).astype(np.uint8)


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
        count = np.count_nonzero(mask == label)
        sats[f'dist_{label}'] = round(count / num_px, 5)
    return sats


def _process_batch(data: Tuple[Hashable, pd.DataFrame],
                   hr_compressed_mask_path: Path,
                   hr_orthofoto_path: Path,
                   hr_harm_path: Path,
                   lr_s2_path: Path,
                   lr_harm_path: Path,) -> Dict:

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
        s2_profile.update(blockxsize=32, blockysize=32, nodata=0)

        # load corresponding orthofoto
        # ortho_base = Path('/data/USERS/shollend/orthophoto/austria_full_allclasses_regridded')
        ortho_base = Path('C:/Users/PC/Coding/cubexpress_austria/local_experiment/local_orthofotos')
        input_path = ortho_base / 'input' / f'input_{id}.tif'
        target_path = ortho_base / 'target' / f'target_{id}.tif'

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

        # All resampling will be conducted with Pytorch Pillow bilinear wrapper as it introduces the least errors
        # https://zuru.tech/blog/the-dangers-behind-image-resizing
        # methods for dimesion filtering is included in resampling_*torch()

        nodata_hr_base = np.all(np.concatenate([mdata, odata], axis=0) != 0, axis=0) * 1  # (512, 512)

        # Harmonising and Matching
        s2_corrs, lr_harms, hr_harms, lr_nodata, hr_nodata = {}, {}, {}, {}, {}
        for s2_name, s2_image in s2_data.items():
            # [4, 3, 2, 8] bands Required -> remapped to [3, 2, 1, 7]
            lr_s2_base = s2_image[[3, 2, 1, 7], :, :] / 10_000

            # create binary mask for filtering and homogenising all image data
            nodata_lr_s2 = np.all(lr_s2_base != 0, axis=0) * 1  # (128, 128)
            nodata_hr_s2 = resample_mask_torch(arr=nodata_lr_s2, scale_factor=4)  # (128, 128) -> (512, 512)

            # stack since arrays are now single band (512, 512)
            nodata_hr = np.all(np.stack([nodata_hr_base, nodata_hr_s2], axis=0) != 0, axis=0) * 1  # (512, 512)
            nodata_lr = resample_mask_torch(arr=nodata_hr, scale_factor=0.25)  # (128, 128)

            # apply masks and normalize
            hr_ortho_norm = odata / 255 * nodata_hr
            lr_s2_norm = lr_s2_base * nodata_lr

            # histogram matching
            # hr_harm_self = utils_histogram.match_histograms(image=hr_ortho_norm, reference=lr_s2, channel_axis=0, ignore_none=True, none_value=normalized_nodata_value)
            # if False:
            #     hr_ortho_norm_small = resample_torch(arr=hr_ortho_norm, scale_factor=0.25)
            #     hr_harm_self_weird = histogram_matching.match_histograms(image=hr_ortho_norm_small,
            #                                                        reference=lr_s2_norm, channel_axis=0)
            #
            #     hr_harm_self = histogram_matching.match_histograms(image=hr_ortho_norm_small.transpose(1, 2, 0),
            #                                                              reference=lr_s2_norm.transpose(1, 2, 0), channel_axis=0)
            #
            #     import matplotlib.pyplot as plt
            #     fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            #     # Plot the 128x128 array
            #     ax[0].imshow(hr_harm_self_weird[:, :3, :].transpose(2, 0, 1), cmap='viridis')  # You can change colormap as needed
            #     ax[0].set_title("Weird")
            #     ax[0].axis('off')  # Hide axis for clarity
            #
            #     # Plot the 512x512 array
            #     ax[1].imshow(hr_harm_self[:3].transpose(1, 2, 0), cmap='viridis')
            #     ax[1].set_title("Normal")
            #     ax[1].axis('off')  # Hide axis for clarity
            #
            #     # Show the plot
            #     plt.tight_layout()
            #     plt.show()
            #
            #     pass
            #
            #     #hr_harm_self = histogram_matching.match_histograms(image=hr_ortho_norm.transpose(1, 2, 0), reference=lr_s2_norm.transpose(1, 2, 0), channel_axis=0)

            hr_harm_self = np.zeros_like(hr_ortho_norm)
            for band in range(hr_ortho_norm.shape[0]):
                hr_harm_self[band] = histogram_matching.match_histograms(image=hr_ortho_norm[band], reference=lr_s2_norm[band])

            # reduce with bilinear interpolation -> (4, 128, 128)
            lr_harm_self = resample_torch(arr=hr_harm_self, scale_factor=0.25)  # (4, 128, 128)

            # Compute block-wise correlation between LR and LRharm
            # the way i have written the correlatin HIGH nan value images are prioitised as they have a large overlap!!!!!!!!
            # corr_self = utils_histogram.own_bandwise_correlation(lr_s2_norm * nodata_lr, lr_harm_self * nodata_lr, none_value=0)
            corr = utils_histogram.fast_block_correlation(lr_s2_norm * nodata_lr, lr_harm_self * nodata_lr, none_value=0)

            # Report the 10th percentile of the correlation (low correlation) without nans
            # low_cor_self = np.nanquantile(corr_self, 0.10)
            low_corr = np.nanquantile(corr, 0.10)

            s2_corrs[s2_name] = round(float(low_corr), 4)
            lr_harms[s2_name] = lr_harm_self
            hr_harms[s2_name] = hr_harm_self
            lr_nodata[s2_name] = nodata_lr
            hr_nodata[s2_name] = nodata_hr

            # test with official changed method (including no data)
            # # hr_ortho_norm_ = np.where(hr_ortho_norm == 0, np.nan, hr_ortho_norm)
            # # lr_s2_ = np.where(lr_s2 == 0, np.nan, lr_s2)
            # hr_harm = utils_histogram.real_match_histograms(image=hr_ortho_norm, reference=lr_s2, channel_axis=0)
            # lr_harm = resize(hr_harm, (4, 128, 128), anti_aliasing=False)
            # corr = utils_histogram.fast_block_correlation(lr_s2, lr_harm, block_size=kernel_size)
            # low_cor_ = np.nanquantile(corr, 0.10)
            # pass

        # Best fitting sentinel2 sample for the current orthophoto
        best_s2_key = max(s2_corrs, key=s2_corrs.get)
        hr_mask = hr_nodata[best_s2_key]

        # apply shared mask
        masked_mdata = mdata * hr_nodata[best_s2_key]
        masked_odata = odata * hr_nodata[best_s2_key]

        ########### ADD METADATA ###########
        batch_statistics['low_corr'] = s2_corrs[best_s2_key]
        batch_statistics['all_corrs'] = s2_corrs
        batch_statistics['all_corrs'] = s2_corrs
        batch_statistics['s2_available'] = True

        # Recalculate mask distribution statistics after masking with nodata
        mask_stats = calculate_mask_statistics(mask=masked_mdata)
        for k, v in mask_stats.items():
            batch_statistics[k] = v

        # derive nodata value on the nodata distribution value from the distribution, rounded to 5 decimals
        if batch_statistics['dist_0'] > 0:
            batch_statistics['contains_nodata'] = True
        else:
            batch_statistics['contains_nodata'] = False

        # add statistical info, except emtpy columns and statistics from the previous mask (before cropping)
        selected_row = batch.loc[batch["s2_download_id"] == best_s2_key]
        row_dict = selected_row.iloc[0].to_dict()
        for k, v in row_dict.items():
            if 'Unnamed' not in k or 'dist_' not in k:
                batch_statistics[k] = v

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
            dst.write((hr_harms[best_s2_key] * hr_nodata[best_s2_key] * 10_000).round().astype(rasterio.uint16))

        # Normal S2 - NODAtA update?
        # file = Path(BASE / "tmp" / f"{best_s2_key}.tif")
        # file.rename(lr_s2_path / f'S2_{new_id}.tif')
        with rasterio.open(lr_s2_path / f'S2_{new_id}.tif', "w", **s2_profile) as dst:
            lr_save_data = s2_data[best_s2_key] * lr_nodata[best_s2_key]
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
            dst.write((lr_harms[best_s2_key] * lr_nodata[best_s2_key] * 10_000).round().astype(rasterio.uint16))

        # remove unwanted sentinel images
        for i, row in batch.iterrows():
            file = Path(BASE / "tmp" / f"{row['s2_download_id']}.tif")
            file.unlink()

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
        df_filtered.to_csv('/home/shollend/shares/users/master/metadata/cubexpress/merged_ALL_S2_filter_wmetadata.csv')
    else:
        # df_filtered: pd.DataFrame = pd.read_csv(BASE / "/home/shollend/shares/users/master/metadata/cubexpress/merged_ALL_S2_filter_wmetadata.csv")
        df_filtered: pd.DataFrame = pd.read_csv(BASE / "sample_s2_wmeta.csv")

    # ortho path
    # /data/USERS/shollend/orthophoto/austria_full_allclasses_regridded
    # table path
    # /home/shollend/shares/users/master/metadata/cubexpress/merged_ALL_S2_filter.csv
    # base
    # /data/USERS/shollend/combined_download

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
    tstart = time.time()
    # Create a multiprocessing pool
    rows = [(idx, batch) for idx, batch in df_batches]

    res = []
    for row in tqdm(rows[20:]):
        res.append(_process_batch(data=row,
                                  hr_compressed_mask_path=out_ortho_target,
                                  hr_orthofoto_path=out_ortho_input,
                                  hr_harm_path=out_ortho_input_harm,
                                  lr_s2_path=out_sentinel2,
                                  lr_harm_path=out_sentinel2_harm,
                                  ))

    # _process_batch_partial = partial(_process_batch,
    #                                  hr_compressed_mask_path=out_ortho_target,
    #                                  hr_orthofoto_path=out_ortho_input,
    #                                  hr_harm_path=out_ortho_input_harm,
    #                                  lr_s2_path=out_sentinel2,
    #                                  lr_harm_path=out_sentinel2_harm,)
    #
    # # Run parallel processing
    # with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    #     res = list(tqdm(executor.map(_process_batch_partial, rows[20:25]), total=len(rows)))

    out_df = pd.DataFrame.from_records(res)
    out_df.to_csv(BASE / 's2_ortho_download_data.csv')

    tstop = time.time()
    print(f'Script took {round(tstop-tstart, 2)}s')

    for i, b in enumerate(res):
        if not b:
            print(f'Failed to process batch index/id: {i}')
