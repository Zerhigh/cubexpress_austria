import datetime as dt

import ee
import pandas as pd
import geopandas as gpd
from typing import Tuple, List, Optional, Dict
import utm
import os
import time
from tqdm import tqdm
import cubexpress
from multiprocessing import Pool
from typing import List, Optional
from pathlib import Path


try:
    ee.Initialize()
except Exception:
    ee.Authenticate(auth_mode="notebook")
    ee.Initialize()

CDF_TRESHHOLD = 0.9

def query_utm_crs_info(
    lon: float,
    lat: float
) -> Tuple[float, float, str]:
    """
    Convert a pair of latitude/longitude coordinates to UTM coordinates
    and generate the corresponding EPSG code.

    Args:
        lon (float): Longitude in decimal degrees.
        lat (float): Latitude in decimal degrees.

    Returns:
        Tuple[float, float, str]:
            - UTM x coordinate
            - UTM y coordinate
            - UTM EPSG code (e.g., 'EPSG:32618' for northern hemisphere)
    """
    x, y, zone_number, _ = utm.from_latlon(lat, lon)
    # Determine whether we are in the northern or southern hemisphere
    zone_epsg = f"326{zone_number:02d}" if lat >= 0 else f"327{zone_number:02d}"
    return x, y, f"EPSG:{zone_epsg}"


def image_to_feature(img: ee.Image) -> ee.Feature:
    """
    Convert an Earth Engine Image footprint (system:footprint) into a Feature
    that retains the image properties.

    Args:
        img (ee.Image): Earth Engine Image with a 'system:footprint' property.

    Returns:
        ee.Feature: A feature representing the image footprint,
                    carrying the image's metadata.
    """
    ring = ee.Geometry(img.get("system:footprint"))
    poly = ee.Geometry.Polygon([ring.coordinates()])
    return ee.Feature(poly, img.toDictionary())


def calculate_timeframe(start: str, stop: str, min_days: int = 100) -> Tuple[str, str]:
    dt_start, dt_stop = dt.datetime.strptime(start, '%Y-%m-%d'), dt.datetime.strptime(stop, '%Y-%m-%d')
    to_time_diff = dt.timedelta(days=min_days)

    if dt_stop - dt_start < to_time_diff:
        mean_day = dt_start + (dt_stop - dt_start) / 2
        start_ = mean_day - to_time_diff / 2
        stop_ = mean_day + to_time_diff / 2
        return start_.strftime('%Y-%m-%d'), stop_.strftime('%Y-%m-%d')
    else:
        return start, stop


def _parallel(row: pd.Series) -> Tuple[str, pd.DataFrame | None]:
    # Derive UTM CRS based on point location
    _, _, crs = query_utm_crs_info(row.lon, row.lat)

    # Create an Earth Engine point geometry
    center: ee.Geometry = ee.Geometry.Point([row.lon, row.lat])

    start, stop = calculate_timeframe(row["start_date"], row["end_date"])

    # Create an ImageCollection filtered by date, bounding area, and band
    ic: ee.ImageCollection = (
        ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
        .filterDate(start, stop)
        .filterBounds(center)
        .select("cs_cdf")
    )

    # Convert each image footprint in the collection to a feature
    foot_fc: ee.FeatureCollection = ic.map(image_to_feature)

    # Create a bounding square (640m buffer) in UTM, then reproject to EPSG:4326
    square: ee.Geometry = (
        center
        .transform(ee.Projection(crs), 1)
        .buffer(640)
        .bounds()
        .transform(ee.Projection("EPSG:4326"), 1)
    )

    # Filter the feature collection to keep only footprints that fully contain our bounding square
    filtered_fc: ee.FeatureCollection = foot_fc.filter(
        ee.Filter.contains(leftField=".geo", rightValue=square)
    )

    # Match images in the ImageCollection by 'SOURCE_PRODUCT_ID'
    # to those footprints that passed the filter
    cloud_ic_filtered: ee.ImageCollection = ic.filter(
        ee.Filter.inList("SOURCE_PRODUCT_ID", filtered_fc.aggregate_array("SOURCE_PRODUCT_ID"))
    )

    try:
        # Get region data (list of pixel values) for the center geometry at 640m scale
        s2cc_list: List = cloud_ic_filtered.getRegion(geometry=center, scale=640).getInfo()

        # Convert the results into a DataFrame
        df_raw = pd.DataFrame(s2cc_list[1:], columns=s2cc_list[0])
        df_raw["s2_available"] = False

        # Convert times from milliseconds to datetime
        df_raw["time"] = pd.to_datetime(df_raw["time"], unit="ms")
        df_raw["base_date"] = row["Date"]
        df_raw["days_diff"] = (df_raw["time"] - df_raw["base_date"]).dt.days

        # Convert time columns to strings for final export
        df_raw["time"] = df_raw["time"].dt.strftime("%Y-%m-%d")
        df_raw["base_date"] = df_raw["base_date"].dt.strftime("%Y-%m-%d")
        df_raw["abs_days_diff"] = df_raw["days_diff"].abs()

        # Rename certain columns
        df_raw.rename(
            columns={
                "id": "s2_id",
                "longitude": "lon",
                "latitude": "lat",
            },
            inplace=True
        )

        # Overwrite lon/lat with the point's original location
        # (as the getRegion data might alter them slightly)
        df_raw["lon"] = row["lon"]
        df_raw["lat"] = row["lat"]

        # Filter images based on a cloud probability threshold
        df = df_raw[df_raw["cs_cdf"] > CDF_TRESHHOLD].copy()

        # Discard empty or insufficient results
        if df.empty:
            return row['id'], None
        else:

            # Sort by the absolute difference in days and cloud score
            df_final = df.sort_values(by=["cs_cdf", "abs_days_diff"])

            # Select up to 8 best observations
            df_final = df_final.iloc[:8]

            # Append the relevant row attributes (skipping first two and last three columns)
            df_final = df_final.assign(**row[2:])

            # Collect the results in a list
            return row['id'], df_final

    except Exception as e:
        # In case of errors, append None
        print(f'Error: {e}')
        return row['id'], None


def parallel(df: pd.DataFrame) -> list:
    rows = [row for _, row in df.iterrows()]

    with Pool(processes=os.cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(_parallel, rows), total=len(rows), desc="Processing"))

    return results


def sequential(df: pd.DataFrame) -> list:
    results = []
    for _, row in tqdm(df.iterrows()):
        results.append(_parallel(row))

    return results


if __name__ == '__main__':
    BASE = Path('')

    #sampling_path: Path = Path("C:/Users/shollend/data/metadata")
    sampling_path: Path = Path("C:/Users/PC/Desktop/TU/Master/MasterThesis/data/metadata/sampling")
    points: gpd.GeoDataFrame = gpd.read_file(sampling_path / "ALL_S2_points_regular_grid_s2download.gpkg")

    # Define date ranges around the 'Date' column
    points["start_date"] = points["beginLifeS"].dt.strftime('%Y-%m-%d')
    points["end_date"] = points["endLifeSpa"].dt.strftime('%Y-%m-%d')
    points["base_date"] = points["Date"].dt.strftime('%Y-%m-%d')

    #sample = points.sample(n=200)

    print('start')
    start = time.time()
    result = parallel(df=points)
    print(f'took: {round(time.time()-start, 2)}s')

    # Prepare a list to hold all filtered results
    dfs_list: List[Optional[pd.DataFrame]] = []
    id_list_none: List[str] = []

    for i, val in result:
        if val is None:
            id_list_none.append(i)
        else:
            dfs_list.append(val)

    if len(dfs_list) < 1:
        raise TypeError("error, nothing found")
    elif len(id_list_none) > 1:
        print(f'Didnt find anything ({len(id_list_none)}) for {id_list_none}')

    # Combine all dataframes into a single GeoDataFrame
    geo_dataframe = gpd.GeoDataFrame(pd.concat(dfs_list, ignore_index=True), crs="EPSG:4326")
    filtered_sample = points[points['id'].isin(id_list_none)]

    # Save as a GeoPackage (vector file) and CSV
    geo_dataframe.to_file(BASE / "tables" / "ccs_090_ALL_S2_filter.gpkg", driver="GPKG")
    filtered_sample.to_file(BASE / "tables" / "NOT_090_FOUND_S2_filter.gpkg", driver="GPKG")
    geo_dataframe.drop(columns=["geometry"]).to_csv(BASE / "tables" / "ccs_090_ALL_S2_filter.csv", index=False)
