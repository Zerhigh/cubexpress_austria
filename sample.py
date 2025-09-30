 
################################################
############# Generate S2 metadata #############
################################################
import ee
import pandas as pd
import geopandas as gpd
from typing import Tuple, List, Optional
import utm
from tqdm import tqdm
import cubexpress
from typing import List, Optional


from pathlib import Path

# ----------------------------------------------
# Initialize the Earth Engine API
# --------------------------------------------------
try:
    ee.Initialize()
except Exception:
    ee.Authenticate(auth_mode="notebook")
    ee.Initialize()

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

# --------------------------------------------------
# Load and preprocess input points
# --------------------------------------------------

#BASE = Path('drive/MyDrive/Colab Notebooks/')
BASE = Path('')

sampling_path: Path = Path("C:/Users/PC/Desktop/TU/Master/MasterThesis/data/metadata/sampling")
points: gpd.GeoDataFrame = gpd.read_file(sampling_path / "ALL_S2_points_regular_grid_s2download.gpkg")

# Define date ranges around the 'Date' column
points["start_date"] = points["beginLifeS"].dt.strftime('%Y-%m-%d')
points["end_date"] = points["endLifeSpa"].dt.strftime('%Y-%m-%d')
points["base_date"] = points["Date"].dt.strftime('%Y-%m-%d')

# Prepare a list to hold all filtered results
dfs_list: List[Optional[pd.DataFrame]] = []

# --------------------------------------------------
# Main loop: process each point to filter S2 (Sentinel-2) images
# --------------------------------------------------
for i, row in tqdm(points.iterrows()):
    # Derive UTM CRS based on point location
    _, _, crs = query_utm_crs_info(row.lon, row.lat)

    # Create an Earth Engine point geometry
    center: ee.Geometry = ee.Geometry.Point([row.lon, row.lat])

    # Create an ImageCollection filtered by date, bounding area, and band
    ic: ee.ImageCollection = (
        ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
        .filterDate(row["start_date"], row["end_date"])
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

        # Filter images based on a cloud probability threshold
        df = df_raw[df_raw["cs_cdf"] > 0.8].copy()

        # Discard empty or insufficient results
        if df.empty:
            pass
        elif len(df) < 8:
            pass
        else:
            # Convert times from milliseconds to datetime
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            df["base_date"] = row["Date"]
            df["days_diff"] = (df["time"] - df["base_date"]).dt.days

            # Convert time columns to strings for final export
            df["time"] = df["time"].dt.strftime("%Y-%m-%d")
            df["base_date"] = df["base_date"].dt.strftime("%Y-%m-%d")
            df["abs_days_diff"] = df["days_diff"].abs()

            # Rename certain columns
            df.rename(
                columns={
                    "id": "s2_id",
                    "longitude": "lon",
                    "latitude": "lat",
                },
                inplace=True
            )

            # Overwrite lon/lat with the point's original location 
            # (as the getRegion data might alter them slightly)
            df["lon"] = row["lon"]
            df["lat"] = row["lat"]

            # Sort by the absolute difference in days and cloud score
            df_final = df.sort_values(by=["abs_days_diff", "cs_cdf"])

            # Select up to 8 best observations
            df_final = df_final.iloc[:8]

            # Append the relevant row attributes (skipping first two and last three columns)
            df_final = df_final.assign(**row[2:-3])

            # Collect the results in a list
            dfs_list.append(df_final)

        if i % 10000 == 0 and i != 0:
            geo_dataframe_temp = gpd.GeoDataFrame(pd.concat(dfs_list, ignore_index=True), crs="EPSG:4326")

            # Save as a GeoPackage (vector file) and CSV
            geo_dataframe_temp.to_file(BASE / f"tables/stratified_ALL_S2_points_wdate_filter_{i}.gpkg", driver="GPKG")
            geo_dataframe_temp.drop(columns=["geometry"]).to_csv(BASE / f"tables/stratified_ALL_S2_points_wdate_filter_{i}.csv",
                                                            index=False)
    except Exception as e:
        # In case of errors, append None
        dfs_list.append(None)

# --------------------------------------------------
# Merge results and export
# --------------------------------------------------
# Combine all dataframes into a single GeoDataFrame
geo_dataframe = gpd.GeoDataFrame(pd.concat(dfs_list, ignore_index=True), crs="EPSG:4326")

# Save as a GeoPackage (vector file) and CSV
geo_dataframe.to_file(BASE / "tables/stratified_ALL_S2_points_wdate_filter.gpkg", driver="GPKG")
geo_dataframe.drop(columns=["geometry"]).to_csv(BASE / "tables/stratified_ALL_S2_points_wdate_filter.csv", index=False)


#######################################
############# Download S2 #############
#######################################

# --------------------------------------------------
# Initialize Earth Engine
# --------------------------------------------------
# try:
#     ee.Initialize()
# except Exception:
#     ee.Authenticate(auth_mode="colab")
#     ee.Initialize()

# --------------------------------------------------
# Load the table containing Sentinel-2 metadata
# --------------------------------------------------
table: pd.DataFrame = pd.read_csv(BASE / "tables/stratified_ALL_S2_points_wdate_filter.csv")

# Selecting the best of the best by ID
df_sorted = table.sort_values(
    by = ["id", "abs_days_diff", "cs_cdf"],
    ascending=[True, True, False]
)

df_selected = df_sorted.groupby("id").first().reset_index()

# You can change this because it's just a way to put new ids
df_selected["s2_download_id"] = [
    f"S2_U_{i:05d}" for i in range(len(df_selected))
]

# Filter out unique Sentinel-2 IDs
filter_ids: List[str] = df_selected["s2_id"].unique().tolist()

# --------------------------------------------------
# Check if IDs are in the SR (Surface Reflectance) collection
# --------------------------------------------------
ic_sr = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filter(ee.Filter.inList("system:index", filter_ids))
)
valid_sr_ids: List[str] = ic_sr.aggregate_array("system:index").getInfo()

# Collect the remaining IDs not found in the SR collection
valid_ids: List[str] = [
    item for item in filter_ids if item not in valid_sr_ids
]

def build_sentinel2_path(s2_id: str) -> str:
    """
    Return the appropriate Sentinel-2 image path based on ID availability.

    Args:
        s2_id (str): The Sentinel-2 image ID (system:index).

    Returns:
        str: The full Sentinel-2 asset path.
    """
    if s2_id in valid_sr_ids:
        return f"COPERNICUS/S2_SR_HARMONIZED/{s2_id}"
    elif s2_id in valid_ids:
        return f"COPERNICUS/S2_HARMONIZED/{s2_id}"
    else:
        return f"UNKNOWN/{s2_id}"

# Build full Sentinel-2 paths in the table
df_selected["s2_full_id"] = df_selected["s2_id"].apply(build_sentinel2_path)

# Filter the dataframe for SR images only
df_filtered: pd.DataFrame = df_selected[
    df_selected["s2_full_id"].str.startswith("COPERNICUS/S2_SR_HARMONIZED/")
].copy()

df_filtered.to_csv(BASE / 'tables/download_stratified_ALL_S2_points_wdate_filter.csv')

# --------------------------------------------------
# Download with cubexpress
# --------------------------------------------------

print('downloading sentinel2')
for i, row in tqdm(df_filtered.iterrows()):
    # Prepare the raster transform
    geotransform = cubexpress.lonlat2rt(
        lon=row["lon"],
        lat=row["lat"],
        edge_size=128,
        scale=10
    )

    # Build a single Request
    request = cubexpress.Request(
        id=row["s2_download_id"],
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
        output_path=BASE / "output",  # directory for output
        nworkers=4,              # parallel workers
        max_deep_level=5         # maximum concurrency with Earth Engine
    )
