import pandas as pd
import xarray as xr
import numpy as np
import glob, os, sys
from pathlib import Path
from p_tqdm import p_map
import seaborn as sns
import matplotlib.pyplot as plt
from dscim.menu.simple_storage import EconVars
import dask
from dask.distributed import Client, progress


# TO-DO: need to update path and put paths in function parameters
def resave_labor_histclim(i):
    def prep(ds, i=i):
        return ds.sel(gcm=i).drop("gcm")

    ec = EconVars(
        path_econ="/shares/gcp/integration/float32/dscim_input_data/econvars/"
    )

    delta_paths = Path(
        f"/shares/gcp/integration/float32/input_data_histclim/labor_data/new_mc/"
    ).glob("rebased_wage-levels_batch*.zarr")

    damages = xr.open_mfdataset(delta_paths, preprocess=prep, parallel=True)

    damages = damages.expand_dims({"gcm": [str(i)]})

    damages = damages.chunk(
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

    to_store = damages.copy()
    for var in to_store.variables:
        to_store[var].encoding.clear()

    if i == "surrogate_GFDL-ESM2G_06":
        to_store.to_zarr(
            f"/shares/gcp/integration/float32/input_data_histclim/labor_data/new_mc.zarr",
            consolidated=True,
            mode="w",
        )
    else:
        to_store.to_zarr(
            f"/shares/gcp/integration/float32/input_data_histclim/labor_data/new_mc.zarr",
            consolidated=True,
            append_dim="gcm",
        )

    damages.close()
    to_store.close()


def resave_energy_histclim(i):
    def prep(ds, i=i):
        return ds.sel(gcm=i).drop("gcm")

    ec = EconVars(
        path_econ="/shares/gcp/integration/float32/dscim_input_data/econvars/"
    )

    delta_paths = Path(
        f"/shares/gcp/integration/float32/input_data_histclim/energy_data/hybrid_price/"
    ).glob("rebased_batch*.zarr")

    damages = xr.open_mfdataset(delta_paths, preprocess=prep, parallel=True)

    damages = damages.expand_dims({"gcm": [str(i)]})

    damages = damages.chunk(
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

    to_store = damages.copy()
    for var in to_store.variables:
        to_store[var].encoding.clear()

    if i == "surrogate_GFDL-ESM2G_06":
        to_store.to_zarr(
            f"/shares/gcp/integration/float32/input_data_histclim/energy_data/hybrid_price.zarr",
            consolidated=True,
            mode="w",
        )
    else:
        to_store.to_zarr(
            f"/shares/gcp/integration/float32/input_data_histclim/energy_data/hybrid_price.zarr",
            consolidated=True,
            append_dim="gcm",
        )

    damages.close()
    to_store.close()


def rechunk_AMEL_indiv_sectors():

    gcm = [
        "ACCESS1-0",
        "CCSM4",
        "GFDL-CM3",
        "IPSL-CM5A-LR",
        "MIROC-ESM-CHEM",
        "bcc-csm1-1",
        "CESM1-BGC",
        "GFDL-ESM2G",
        "IPSL-CM5A-MR",
        "MPI-ESM-LR",
        "BNU-ESM",
        "CNRM-CM5",
        "GFDL-ESM2M",
        "MIROC5",
        "MPI-ESM-MR",
        "CanESM2",
        "CSIRO-Mk3-6-0",
        "inmcm4",
        "MIROC-ESM",
        "MRI-CGCM3",
        "NorESM1-M",
        "surrogate_GFDL-CM3_89",
        "surrogate_GFDL-ESM2G_11",
        "surrogate_CanESM2_99",
        "surrogate_GFDL-ESM2G_01",
        "surrogate_MRI-CGCM3_11",
        "surrogate_CanESM2_89",
        "surrogate_GFDL-CM3_94",
        "surrogate_MRI-CGCM3_01",
        "surrogate_CanESM2_94",
        "surrogate_GFDL-CM3_99",
        "surrogate_MRI-CGCM3_06",
        "surrogate_GFDL-ESM2G_06",
    ]

    gcm.reverse()

    # Energy
    for i, g in enumerate(gcm):
        print(i)
        resave_energy_histclim(g)

    # Labour
    for i, g in enumerate(gcm):
        print(i)
        resave_labor_histclim(g)

    # Agriculture - runs slightly differently : use a Dask client

    dask.config.set(
        {
            "distributed.worker.memory.target": 0.7,
            "distributed.worker.memory.spill": 0.8,
            "distributed.worker.memory.pause": 0.9,
        }
    )

    client = Client(n_workers=10, memory_limit="30G", threads_per_worker=1)

    chunkies = {
        "rcp": 1,
        "region": 24378,
        "gcm": 1,
        "year": 10,
        "model": 1,
        "ssp": 1,
        "batch": 15,
    }

    delta = xr.open_mfdataset(
        Path(
            "/shares/gcp/integration/float32/input_data_histclim/ag_data/gdp_weights_delta"
        ).rglob("damages_batch*.nc4")
    )
    delta = (
        delta.chunk(chunkies)
        .squeeze()
        .drop("variable")
        .reset_coords("variable", drop=True)
    )
    delta = xr.where(np.isinf(delta), np.nan, delta)
    ds = xr.Dataset({"delta_reallocation": delta.wc_reallocation})

    ds.to_zarr(
        "/mnt/sacagawea_shares/gcp/integration/float32/input_data_histclim/ag_data/gdp_weights_ag_histclim-delta.zarr",
        mode="w",
        consolidated=True,
    )

    client.restart()

    histclim = xr.open_mfdataset(
        Path(
            "/shares/gcp/integration/float32/input_data_histclim/ag_data/gdp_weights_histclim"
        ).rglob("damages_batch*.nc4")
    )
    histclim = (
        histclim.chunk(chunkies)
        .squeeze()
        .drop("variable")
        .reset_coords("variable", drop=True)
    )
    histclim = xr.where(np.isinf(histclim), np.nan, histclim)
    ds = xr.Dataset(
        {
            "histclim_reallocation": histclim.wc_reallocation,
        }
    )

    ds.to_zarr(
        "/mnt/sacagawea_shares/gcp/integration/float32/input_data_histclim/ag_data/gdp_weights_ag_histclim-delta.zarr",
        mode="a",
        consolidated=True,
    )
