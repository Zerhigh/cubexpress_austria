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


BASE = Path('')
df_filtered: pd.DataFrame = pd.read_csv(BASE / "tables" / "ccs_090_ALL_S2_filter_sample200_withmetadata.csv")

ortho_urls = df_filtered['']

pass