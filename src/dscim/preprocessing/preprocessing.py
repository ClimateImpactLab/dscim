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
