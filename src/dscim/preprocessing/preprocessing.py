from math import ceil
from pathlib import Path
import os
import sys

USER = os.getenv("USER")
import dask
import dask.array as da
import dask.config
import numpy as np
import xarray as xr
from dask.distributed import Client, progress
from dscim.utils.functions import ce_func, mean_func
import yaml
import time
import argparse


def ce_from_chunk(
    chunk,
    filepath,
    reduction,
    bottom_code,
    histclim,
    delta,
    recipe,
    eta,
    zero,
    socioec,
    ce_batch_coords,
):

    year = chunk.year.values
    ssp = chunk.ssp.values
    model = chunk.model.values

    gdppc = (
        xr.open_zarr(socioec, chunks=None)
        .sel(
            year=year, ssp=ssp, model=model, region=ce_batch_coords["region"], drop=True
        )
        .gdppc
    )

    if reduction == "no_cc":
        if zero:
            chunk[histclim] = xr.where(chunk[histclim] == 0, 0, 0)
        calculation = gdppc + chunk[histclim].mean("batch") - chunk[histclim]
    elif reduction == "cc":
        calculation = gdppc - chunk[delta]
    else:
        raise NotImplementedError("Pass 'cc' or 'no_cc' to reduction.")

    if recipe == "adding_up":
        result = mean_func(
            np.maximum(
                calculation,
                bottom_code,
            ),
            "batch",
        )
    elif recipe == "risk_aversion":
        result = ce_func(
            np.maximum(
                calculation,
                bottom_code,
            ),
            "batch",
            eta=eta,
        )

    return result


def reduce_damages(
    recipe,
    reduction,
    eta,
    sector,
    config,
    socioec,
    bottom_coding_gdppc=39.39265060424805,
    zero=False,
):
    if recipe == "adding_up":
        assert (
            eta is None
        ), "Adding up does not take an eta argument. Please set to None."
    # client = Client(n_workers=35, memory_limit="9G", threads_per_worker=1)

    with open(config, "r") as stream:
        c = yaml.safe_load(stream)
        params = c["sectors"][sector]

    damages = Path(params["sector_path"])
    histclim = params["histclim"]
    delta = params["delta"]
    outpath = f"{c['paths']['reduced_damages_library']}/{sector}"

    with xr.open_zarr(damages, chunks=None)[histclim] as ds:
        with xr.open_zarr(socioec, chunks=None) as gdppc:

            assert (
                xr.open_zarr(damages).chunks["batch"][0] == 15
            ), "'batch' dim on damages does not have chunksize of 15. Please rechunk."

            ce_batch_dims = [i for i in gdppc.dims] + [
                i for i in ds.dims if i not in gdppc.dims and i != "batch"
            ]
            ce_batch_coords = {c: ds[c].values for c in ce_batch_dims}
            ce_batch_coords["region"] = [
                i for i in gdppc.region.values if i in ce_batch_coords["region"]
            ]
            ce_shapes = [len(ce_batch_coords[c]) for c in ce_batch_dims]
            ce_chunks = [xr.open_zarr(damages).chunks[c][0] for c in ce_batch_dims]

    template = xr.DataArray(
        da.empty(ce_shapes, chunks=ce_chunks),
        dims=ce_batch_dims,
        coords=ce_batch_coords,
    )

    other = xr.open_zarr(damages)

    out = other.map_blocks(
        ce_from_chunk,
        kwargs=dict(
            filepath=damages,
            reduction=reduction,
            bottom_code=bottom_coding_gdppc,
            histclim=histclim,
            delta=delta,
            eta=eta,
            recipe=recipe,
            zero=zero,
            socioec=socioec,
            ce_batch_coords=ce_batch_coords,
        ),
        template=template,
    )

    out = out.astype(np.float32).rename(reduction).to_dataset()

    out.attrs["bottom code"] = bottom_coding_gdppc
    out.attrs["histclim=0"] = zero
    out.attrs["filepath"] = str(damages)

    if recipe == "adding_up":
        out.to_zarr(
            f"{outpath}/{recipe}_{reduction}.zarr",
            consolidated=True,
            mode="w",
        )
    elif recipe == "risk_aversion":
        out.attrs["eta"] = eta
        out.to_zarr(
            f"{outpath}/{recipe}_{reduction}_eta{eta}.zarr",
            consolidated=True,
            mode="w",
        )


def reformat_climate_files():
    from dscim.preprocessing.climate.reformat import (
        convert_old_to_newformat_AR,
        stack_gases,
    )

    # convert AR6 files
    bd = "/shares/gcp/integration/float32/dscim_input_data/climate/AR6"
    pathdt = {
        "median": f"{bd}/ar6_fair162_medianparams_control_pulse_2020-2080_10yrincrements_conc_rf_temp_lambdaeff_emissions-driven_2naturalfix_v4.0_Jan212022.nc",
        "sims": f"{bd}/ar6_fair162_control_pulse_2020-2030-2040-2050-2060-2070-2080_emis_conc_rf_temp_lambdaeff_emissions-driven_naturalfix_v4.0_Jan212022.nc",
    }

    newds = convert_old_to_newformat_AR(
        pathdt,
        gas="CO2_Fossil",
        var="temperature",
    )

    newds.to_netcdf(
        f"{bd}/ar6_fair162_sim_and_medianparams_control_pulse_2030-2040-2050-2060-2070-2080_emis_conc_rf_temp_lambdaeff_emissions-driven_naturalfix_v4.0_Jan212022.nc"
    )

    # convert RFF files
    gases = {"CO2_Fossil": "Feb072022", "CH4": "Feb072022", "N2O": "Feb072022"}
    stack_gases(gas_dict=gases)


def sum_AMEL(
    sectors,
    config,
    AMEL,
):

    # load config
    with open(config, "r") as stream:
        loaded_config = yaml.safe_load(stream)
        params = loaded_config["sectors"]

    output = params[AMEL]["sector_path"]

    # save summed variables to zarr one by one
    for i, var in enumerate(["delta", "histclim"]):

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

        for v in summed.variables:
            summed[v].encoding.clear()

        if i == 0:
            summed.to_zarr(output, consolidated=True, mode="w")
        else:
            summed.to_zarr(output, consolidated=True, mode="a")


def subset_USA_reduced_damages(
    sector,
    reduction,
    recipe,
    eta,
    input_path,
):

    if recipe == "adding_up":
        ds = xr.open_zarr(
            f"{input_path}/{sector}/{recipe}_{reduction}.zarr",
        )
    elif recipe == "risk_aversion":
        ds = xr.open_zarr(
            f"{input_path}/{sector}/{recipe}_{reduction}_eta{eta}.zarr",
        )

    subset = ds.sel(region=[i for i in ds.region.values if "USA" in i])

    for var in subset.variables:
        subset[var].encoding.clear()

    if recipe == "adding_up":
        subset.to_zarr(
            f"{input_path}/{sector}_USA/{recipe}_{reduction}.zarr",
            consolidated=True,
            mode="w",
        )
    elif recipe == "risk_aversion":
        subset.to_zarr(
            f"{input_path}/{sector}_USA/{recipe}_{reduction}_eta{eta}.zarr",
            consolidated=True,
            mode="w",
        )


def subset_USA_ssp_econ(
    in_path,
    out_path,
):

    zarr = xr.open_zarr(
        in_path,
        consolidated=True,
    )

    zarr = zarr.sel(region=[i for i in zarr.region.values if "USA" in i])

    for var in zarr.variables:
        zarr[var].encoding.clear()

    zarr.to_zarr(
        out_path,
        consolidated=True,
        mode="w",
    )


def clip_damages(
    config,
    sector,
    econ_path="/shares/gcp/integration/float32/dscim_input_data/econvars/zarrs/integration-econ-bc39.zarr",
):
    """This function is no longer in use.
    To operationalize, make sure to get a Dask client running with the following code:

    # set up dask
    dask.config.set(
        {
            "distributed.worker.memory.target": 0.7,
            "distributed.worker.memory.spill": 0.8,
            "distributed.worker.memory.pause": 0.9,
        }
    )

    client = Client(n_workers=40, memory_limit="9G", threads_per_worker=1)
    """

    # load config
    with open(config, "r") as stream:
        loaded_config = yaml.safe_load(stream)
        params = loaded_config["sectors"][sector]

    # get sector paths and variable names
    sector_path = Path(params["sector_path"])
    histclim = params["histclim"]
    delta = params["delta"]

    with xr.open_zarr(sector_path, chunks=None)[delta] as ds:
        with xr.open_zarr(econ_path, chunks=None) as gdppc:

            ce_batch_dims = [i for i in ds.dims]
            ce_batch_coords = {c: ds[c].values for c in ce_batch_dims}
            ce_batch_coords["region"] = [
                i for i in ds.region.values if i in gdppc.region.values
            ]
            ce_shapes = [len(ce_batch_coords[c]) for c in ce_batch_dims]
            ce_chunks = [xr.open_zarr(sector_path).chunks[c][0] for c in ce_batch_dims]
            print(ce_chunks)

    template = xr.DataArray(
        da.empty(ce_shapes, chunks=ce_chunks),
        dims=ce_batch_dims,
        coords=ce_batch_coords,
    )

    def chunk_func(
        damages,
    ):

        year = damages.year.values
        ssp = damages.ssp.values
        model = damages.model.values
        region = damages.region.values

        gdppc = (
            xr.open_zarr(econ_path, chunks=None)
            .sel(year=year, ssp=ssp, model=model, region=region, drop=True)
            .gdppc
        )

        # get damages as % of GDPpc
        shares = damages / gdppc

        # find the 1st/99th percentile of damage share
        # across batches and regions
        quantile = shares.quantile([0.01, 0.99], ["batch", "region"])

        # find the equivalent damages
        # if damage share is capped to 1st/99th percentile
        quantdams = quantile * gdppc

        # keep damages that are within cutoff,
        # otherwise replace with capped damages
        damages = xr.where(
            (shares <= quantile.sel(quantile=0.99, drop=True)),
            damages,
            quantdams.sel(quantile=0.99, drop=True),
        )

        damages = xr.where(
            (shares >= quantile.sel(quantile=0.01, drop=True)),
            damages,
            quantdams.sel(quantile=0.01, drop=True),
        )

        return damages

    data = xr.open_zarr(sector_path)

    for var in [delta, histclim]:
        out = (
            data[var].map_blocks(chunk_func, template=template).rename(var).to_dataset()
        )

        parent, name = sector_path.parent, sector_path.name
        clipped_name = name.replace(".zarr", "_clipped.zarr")
        outpath = Path(parent).joinpath(clipped_name)

        out.to_zarr(outpath, mode="a", consolidated=True)
