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


BASE = Path('')

sampling_path: Path = Path("C:/Users/PC/Desktop/TU/Master/MasterThesis/data/metadata/sampling")
points: gpd.GeoDataFrame = gpd.read_file(sampling_path / "ALL_S2_points_regular_grid_s2download.gpkg")

# Define date ranges around the 'Date' column
points["start_date"] = points["beginLifeS"].dt.strftime('%Y-%m-%d')
points["end_date"] = points["endLifeSpa"].dt.strftime('%Y-%m-%d')
points["base_date"] = points["Date"].dt.strftime('%Y-%m-%d')

# Prepare a list to hold all filtered results
dfs_list: List[Optional[pd.DataFrame]] = []

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