"""
Run valuations for mortality
"""

import os
import time
from pathlib import Path
import xarray as xr

if __name__ == "__main__":

    hybrids = "/project2/mgreenst/mortality_data/hybrid_damages"
    files = [
        f"{hybrids}/monetized_damages_complete_batch{i}_hybridssp.nc4" for i in [4, 7]
    ]
    for i in files:
        print(i)
        ds = xr.open_dataset(i).load()
        ds = ds.expand_dims("ssp")
        ds.close()
        time.sleep(5)
        ds.to_netcdf(i)
        ds.close()
        time.sleep(5)
