from functools import reduce
import xarray as xr
import os, time, yaml
from pathlib import Path
import numpy as np

USER = os.getenv("USER")
import dask

dask.config.set(**{"array.slicing.split_large_chunks": False})

# set parameters of script
root_dir = "/shares/gcp/integration/float32/input_data_histclim"
mortality_version = 4
sectors = ["agriculture", f"mortality_v{mortality_version}", "energy", "labor"]
config = f"/home/{USER}/repos/dscim-cil/configs/epa_tool_config-histclim_AR6.yaml"
output = f"{root_dir}/AMEL_data/AMEL_m{mortality_version}.zarr"

# load config
with open(config, "r") as stream:
    loaded_config = yaml.safe_load(stream)
    params = loaded_config["sectors"]

# save summed variables to zarr one by one
for var in ["delta", "histclim"]:

    datasets = []

    for sector in sectors:
        print(f"Opening {sector},{params[sector]['sector_path']}")
        ds = xr.open_zarr(params[sector]["sector_path"], consolidated=True)
        ds = ds[params[sector][var]].rename(var)
        ds = xr.where(np.isinf(ds), np.nan, ds)
        datasets.append(ds)

    summed = (
        xr.concat(datasets, dim="variable")
        .sum("variable")
        .rename(f"summed_{var}")
        .astype(np.float32)
        .chunk(
            {
                "batch": 15,
                "ssp": 1,
                "model": 1,
                "rcp": 1,
                "gcm": 1,
                "year": 10,
                "region": 24378,
            }
        )
        .to_dataset()
    )

    summed.attrs["paths"] = str({s: params[s]["sector_path"] for s in sectors})
    summed.attrs["delta"] = str({s: params[s]["delta"] for s in sectors})
    summed.attrs["histclim"] = str({s: params[s]["histclim"] for s in sectors})

    for var in summed.variables:
        summed[var].encoding.clear()

    summed.to_zarr(output, consolidated=True, mode="a")
