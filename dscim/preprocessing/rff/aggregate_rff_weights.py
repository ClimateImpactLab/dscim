import xarray as xr
import pandas as pd
import numpy as np
import os, sys
from numpy.testing import assert_allclose
from p_tqdm import p_map


def clean_simulation(draw):
    ds = pd.read_csv(
        f"/shares/gcp/social/rff/fivebeans3/emulate-fivebean-{draw}.csv",
        skiprows=9,
    )

    for i, name in enumerate(["model", "ssp"]):
        ds[name] = ds["name"].str.split("/", expand=True)[i]
    for i, name in enumerate(["year", "model"]):
        ds[name] = ds["model"].str.split(":", expand=True)[i]

    ds = ds.loc[ds.param != "error"]

    ds["model"] = ds.model.replace({"low": "IIASA GDP", "high": "OECD Env-Growth"})

    ds["rff_sp"] = draw

    ds["year"] = ds.year.astype(int)

    ds = ds.set_index(["model", "ssp", "rff_sp", "year"]).to_xarray()["value"]

    return ds


def clean_error(draw):

    ds = pd.read_csv(
        f"/shares/gcp/social/rff/fivebeans3/emulate-fivebean-{draw}.csv",
        skiprows=9,
    )

    for i, name in enumerate(["iso", "year"]):
        ds[name] = ds["name"].str.split(":", expand=True)[i]

    ds = ds.loc[ds.param == "error"]

    ds["rff_sp"] = draw

    ds["year"] = ds.year.astype(int)

    ds = ds.set_index(["iso", "year", "rff_sp"]).value.to_xarray()

    return ds


datasets = p_map(clean_simulation, range(1, 10001, 1))

concatenated = xr.concat(datasets, "rff_sp").interp(
    {"year": range(2010, 2101, 1)}, method="linear"
)

reweighted = concatenated / concatenated.sum(["model", "ssp"])

# make sure weights sum to 1
assert_allclose(reweighted.sum(["model", "ssp"]).values, 1)

# describe and save file
reweighted = reweighted.to_dataset()
reweighted.attrs["version"] = 3
reweighted.attrs[
    "description"
] = """
This set of emulator weights is generated using this script:
dscim/dscim/preprocessing/rff/aggregate_rff_weights.py
It cleans and aggregates the emulator weights csvs, linearly interpolates them between 5 year intervals, reweights them to sum to 1, and converts to ncdf4 format.
"""

reweighted.to_netcdf(
    f"/shares/gcp/integration/rff/damage_function_weights/damage_function_weights3.nc4",
)

error_datasets = p_map(clean_error, range(1, 10001, 1))
error_concatenated = xr.concat(error_datasets, "rff_sp")

# describe and save file
error_concatenated = error_concatenated.to_dataset()
error_concatenated.attrs["version"] = 3
error_concatenated.attrs[
    "description"
] = """
This set of emulator weight errors is generated using this script:
dscim/dscim/preprocessing/rff/aggregate_rff_weights.py
It cleans and aggregates the emulator weights csvs for error rows only, and converts to ncdf4 format.
"""

error_concatenated.to_netcdf(
    f"/shares/gcp/integration/rff/damage_function_weights/weights_errors.nc4",
)
