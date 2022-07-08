from math import ceil
from pathlib import Path
import os, sys

USER = os.getenv("USER")
import dask
import dask.array as da
import dask.config
import numpy as np
import xarray as xr
from dask.distributed import Client, progress
from dscim.utils.functions import ce_func, mean_func
import yaml, time, argparse


def ce_from_chunk(
    chunk,
    filepath,
    ce_type,
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

    if ce_type == "ce_no_cc":
        if zero == True:
            chunk[histclim] = xr.where(chunk[histclim] == 0, 0, 0)
        calculation = gdppc + chunk[histclim].mean("batch") - chunk[histclim]
    elif ce_type == "ce_cc":
        calculation = gdppc - chunk[delta]
    else:
        raise NotImplementedError("Pass 'ce_cc' or 'ce_no_cc' to ce_type.")

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
    outpath,
    bottom_coding_gdppc=39.39265060424805,
    zero=False,
):

    # client = Client(n_workers=35, memory_limit="9G", threads_per_worker=1)

    with open(config, "r") as stream:
        loaded_config = yaml.safe_load(stream)
        params = loaded_config["sectors"][sector]

    damages = Path(params["sector_path"])
    histclim = params["histclim"]
    delta = params["delta"]

    socioec = Path(
        "/shares/gcp/integration/float32/dscim_input_data/econvars/zarrs/integration-econ-bc39.zarr"
    )

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

    # convert string eta to float
    eta = float(eta)

    other = xr.open_zarr(damages)

    out = other.map_blocks(
        ce_from_chunk,
        kwargs=dict(
            filepath=damages,
            ce_type=reduction,
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

    out.attrs["eta"] = eta
    out.attrs["bottom code"] = bottom_coding_gdppc
    out.attrs["histclim=0"] = zero
    out.attrs["filepath"] = str(damages)

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
        pulseyrs=[2020, 2030, 2040, 2050, 2060, 2070, 2080],
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
    output,
):

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
