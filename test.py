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


def singletimeadd():
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

    BASE = Path('')
    df_filtered: pd.DataFrame = pd.read_csv(BASE / "tables" / "ccs_090_ALL_S2_filter.csv")
    # df_filtered["start_date"] = df_filtered["beginLifeS"].strftime('%Y-%m-%d')
    # df_filtered["end_date"] = df_filtered["endLifeSpa"].strftime('%Y-%m-%d')
    # df_filtered["base_date"] = df_filtered["Date"].strftime('%Y-%m-%d')

    for i, row in tqdm(df_filtered.iterrows()):
        start, stop = calculate_timeframe(row["beginLifeS"], row["endLifeSpa"])
        df_filtered.loc[i, "start_date"], df_filtered.loc[i, "end_date"] = start, stop

    df_filtered_re: pd.DataFrame = pd.read_csv(BASE / "tables" / "redone_ccs_010_ALL_S2_filter.csv")

    df_new = pd.concat([df_filtered, df_filtered_re], axis=0)
    df_new.to_csv(BASE / "tables" / "merged_ALL_S2_filter.csv")
    pass


BASE = Path('')
#df_filtered: pd.DataFrame = pd.read_csv(BASE / "tables" / "merged_ALL_S2_filter.csv")
# sample = df_filtered[:200]
#sample.to_csv(BASE / "local_experiment" / "sample_s2.csv")
df = pd.read_csv(BASE / "local_experiment" / "sample_s2.csv")
groups = [int(i) for i in set(df['id'].values)]
print(groups)

pass