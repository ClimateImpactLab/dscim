import xarray as xr
import pandas as pd
import numpy as np


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
