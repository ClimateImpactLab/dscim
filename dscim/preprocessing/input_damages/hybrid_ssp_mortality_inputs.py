"""
Run valuations for mortality
"""

import os
import time
from pathlib import Path
import xarray as xr

print("testing message: version jun 25")


def hybrid_ssp_mortality_inputs(
    input_path="/project2/mgreenst/mortality_data/hybrid_damages",
):
    # if __name__ == "__main__":

    hybrids = input_path
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
