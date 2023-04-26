import xarray as xr
import pandas as pd
import numpy as np
from dscim.menu.simple_storage import Climate
import os
import sys
import yaml
from p_tqdm import p_map

USER = os.getenv("USER")

# parameters
out_dir = "/shares/gcp/integration/rff/climate/masks/CO2_Fossil"
config = f"/home/{USER}/repos/integration/configs/rff_config.yaml"
quantile_list = [
    [0.001, 0.999],
    [0.005, 0.995],
    [0.01, 0.99],
    [0.05, 0.95],
]

# set list of masks to generate ('gdppc', 'emissions', 'climate')
mask_types = ["gdppc", "emissions", "climate"]


def get_mask(mask_set, quantiles, dim="rff_sp"):

    # quantiles of variable
    quants = mask_set.quantile(quantiles, dim)

    # assign False to values outside quantile bounds, else True
    mask = (
        xr.where(
            (mask_set >= quants.sel(quantile=quantiles[0]))
            & (mask_set <= quants.sel(quantile=quantiles[1])),
            True,
            False,
        )
        .rename(f"q{quantiles[0]}_q{quantiles[1]}")
        .to_dataset()
    )

    return mask


for mask_type in mask_types:

    if mask_type == "emissions":
        # mask on cumulative emissions in 2300
        mask_set = (
            xr.open_dataset(
                "/shares/gcp/integration/rff/climate/rff-sp_emissions_all_gases.nc"
            )
            .emissions.sel(gas="C", drop=True)
            .sum("Year")
            .rename({"simulation": "rff_sp"})
        )
    elif mask_type == "climate":
        # mask on temperature anomalies in 2300
        with open(config) as config_file:
            params = yaml.full_load(config_file)
        mask_set = Climate(**params["climate"]).fair_control.temperature.sel(
            year=2300, drop=True
        )
    elif mask_type == "gdppc":
        # mask on GDP per capita in 2300
        socioec = xr.open_dataset(
            "/shares/gcp/integration/rff/socioeconomics/rff_global_socioeconomics.nc4"
        ).sel(year=2300, region="world", drop=True)
        mask_set = socioec.gdp / socioec.pop
    else:
        raise NotImplementedError(
            "Unknown mask type. Pass 'gdp', 'climate', 'emissions'."
        )

    datasets = p_map(
        get_mask,
        [mask_set for i in range(len(quantile_list))],
        quantile_list,
    )

    xr.combine_by_coords(datasets).to_netcdf(f"{out_dir}/{mask_type}_based_masks.nc4")

    # check correct amount of simulations is getting dropped
    a = xr.combine_by_coords(datasets)
    for var in a.data_vars:
        if "simulation" in a.dims:
            a = a.sel(simulation=5)
        else:
            pass

        subset = a[var]
        falsies = subset.where(~subset, drop=True)
        print(mask_type, var, (len(falsies.rff_sp) / 10000 * 100))
