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


datasets = p_map(clean_simulation, range(1, 10001, 1))

# ask James
concatenated = xr.concat(datasets, "rff_sp").interp(
    {"year": range(2010, 2101, 1)}, method="linear"
)

reweighted = concatenated / concatenated.sum(["model", "ssp"])

# make sure weights sum to 1
assert_allclose(reweighted.sum(["model", "ssp"]).values, 1)

reweighted.to_dataset().to_zarr(
    f"/shares/gcp/integration/rff/damage_function_weights/damage_function_weights3.zarr",
    consolidated=True,
)
