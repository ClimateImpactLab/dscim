import xarray as xr
import pandas as pd
import numpy as np
import os
import sys

USER = os.getenv("USER")
from p_tqdm import p_map


def stack(file, itr):

    ds = xr.open_dataset(file)
    ds = ds.rename({"simulation": "rff_sp"}).expand_dims({"simulation": [itr]})

    return ds


print("Stacking climate iterations...")
datasets = p_map(
    stack,
    [
        f"/shares/gcp/integration/rff/climate/ar6_rff_iter{i}_fair162_CO2_Fossil_control_pulse_2020_temp_v5.01_newformat.nc"
        for i in range(1, 6)
    ],
    range(1, 6),
)

xr.concat(datasets, "simulation").to_netcdf(
    "/shares/gcp/integration/rff/climate/ar6_rff_iter1-5_fair162_CO2_Fossil_control_pulse_2020_temp_v5.01_newformat.nc"
)

print("Stacking climate iteration masks...")
datasets = p_map(
    stack,
    [
        f"/shares/gcp/integration/rff/climate/masks/ar6_rff_iter{i}_fair162_CO2_Fossil_control_pulse_2020_temp_v5.01_newformat.nc"
        for i in range(1, 6)
    ],
    range(1, 6),
)

xr.concat(datasets, "simulation").to_netcdf(
    "/shares/gcp/integration/rff/climate/masks/ar6_rff_iter1-5_fair162_CO2_Fossil_control_pulse_2020_temp_v5.01_newformat.nc"
)
